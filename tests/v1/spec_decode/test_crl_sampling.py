# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CRL drafting at temperature > 0: the proposal the walk samples is the proposal the verifier checks.

``CRLSpeculator`` only builds the score tables (``[R, N, K, K]``: ``table[r, k, i, b]`` scores candidate ``b`` of row ``k`` given
that the row before drafted its candidate ``i``; predecessors beyond ``crl_pred_topk`` carry the row's own logits). Everything
after that -- the Gumbel walk, the cache of the realized proposal, the rejection sampler -- is the DFlash 2 machinery, shared with
every other drafter. These tests run those three real kernels on tables with CRL's structure and check, against exact
references, that

* at temperature 0 the walk is the argmax walk of the CRL spec, with or without proposal logits (greedy / probabilistic drafting),
  and a temperature-0 request next to temperature-1 requests is untouched;
* at temperature > 0 a drafted token is distributed as softmax(score row of its *realized* predecessor / T) over the candidates,
  and the cache holds exactly that row (pre-temperature scores on the candidates, -inf elsewhere);
* the full round (walk + cache + rejection sampler) emits the target's distribution: the first two emitted tokens of a Markov
  target follow ``p(t1) p(t2 | t1)``, for greedy and probabilistic drafting, with many predecessors outside the top ``crl_pred_topk``;
* a cache sharper than the proposal actually sampled makes that check fail (the check can see a lossy proposal); a cache holding the
  wrong predecessor's row fails the exact comparison of the cache with the realized row.

No GPU is needed: without CUDA the Triton kernels run in the interpreter (``TRITON_INTERPRET=1``); the vLLM Triton shim is told to
use the real ``triton`` first. Run this file on its own (``python -m pytest tests/v1/spec_decode/test_crl_sampling.py``).
"""

import math
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"  # (read when triton is first imported)

pytest.importorskip("triton")

if not torch.cuda.is_available():
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    import vllm.triton_utils as _triton_utils

    _triton_utils.triton, _triton_utils.tl, _triton_utils.HAS_TRITON = (
        triton,
        tl,
        True,
    )

    class _InterpretedLibdevice:
        """libdevice's functions are extern stubs the interpreter does not run; the rejection sampler needs log1p."""

        log1p = staticmethod(lambda x: tl.log(1.0 + x))

        def __getattr__(self, name):
            return getattr(tldevice, name)

    _triton_utils.tldevice = _InterpretedLibdevice()

from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as dflash2_speculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

if not torch.cuda.is_available():
    from triton.runtime.interpreter import InterpretedFunction

    if not isinstance(dflash2_speculator._selector_walk_kernel, InterpretedFunction):
        pytest.skip(
            "the Triton kernels were imported before the interpreter was switched on: run this file on its own",
            allow_module_level=True,
        )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# requests per distribution check: the Triton interpreter costs ~0.05 s per request for the walk and ~0.15 s for a whole round
# (set CRL_SAMPLING_TRIALS to scale a CPU run up; the checks are only as sharp as the sample)
_DEFAULT_TRIALS = 20000 if torch.cuda.is_available() else 1200
TRIALS = int(os.environ.get("CRL_SAMPLING_TRIALS", _DEFAULT_TRIALS))
K = 5  # candidates per row (selector_top_k)
KA = 2  # crl_pred_topk: predecessors with a conditional table row
N = 3  # draft rows
VOCAB = 8


# ---------------------------------------------------------------------------------------------------------------------------
# tables with the structure CRLSpeculator._score_candidates builds
# ---------------------------------------------------------------------------------------------------------------------------
def make_tables(num_contexts: int, gen: torch.Generator):
    """Per context (= anchor token): candidates ``[C, N, K]`` (top-K of the row's own logits, best first), the row's own
    candidate logits ``[C, N, K]`` and the CRL tables ``[C, N, K, K]``.

    ``table[c, k, i, :]`` for ``i < KA`` is a re-scored row (unary + noise: the predecessor changes the ranking); for ``i >= KA``
    it is the row's own logits -- the fallback. Row 0 is conditioned on the anchor token, so all of its KA conditional rows are
    the same vector.
    """
    unary_full = torch.randn(num_contexts, N, VOCAB, generator=gen) * 1.5
    unary, cand = unary_full.topk(K, dim=-1)
    table = unary[:, :, None, :].expand(-1, -1, K, -1).clone()
    conditional = unary[:, :, None, :] + torch.randn(num_contexts, N, KA, K, generator=gen) * 1.2
    conditional[:, 0] = conditional[:, 0, :1].expand(-1, KA, -1)
    table[:, :, :KA, :] = conditional
    return cand.to(DEVICE), unary.to(DEVICE), table.to(DEVICE)


def walk(cand, table, temperature, seeds, sample_pos, probabilistic):
    """Run CandidateSampler.sample for R requests; ``cand`` [R, N, K], ``table`` [R, N, K, K], ``sample_pos`` [R, N]."""
    r = cand.shape[0]
    sampler = CandidateSampler(r, N, K, DEVICE)
    tokens = torch.zeros(r, N, dtype=torch.int64, device=DEVICE)
    cache = torch.full((r, N, VOCAB), float("-inf"), device=DEVICE) if probabilistic else None
    sampler.sample(
        cand,
        table.contiguous(),
        r,
        sample_pos.flatten(),
        torch.arange(r, dtype=torch.int64, device=DEVICE).repeat_interleave(N),
        temperature,
        seeds,
        tokens,
        cache,
        False,
    )
    return tokens, cache, sampler


def reference_walk(cand, table):
    """The CRL spec's argmax walk (Impl 27.3), written out: row k scores its candidates given the predecessor's index."""
    r = cand.shape[0]
    out = torch.zeros(r, N, dtype=torch.int64)
    index = torch.zeros(r, N, dtype=torch.int64)
    for q in range(r):
        prev = 0
        for k in range(N):
            i = int(table[q, k, prev].argmax())
            out[q, k] = cand[q, k, i]
            index[q, k] = i
            prev = i
    return out, index


# ---------------------------------------------------------------------------------------------------------------------------
# temperature 0
# ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("probabilistic", [False, True])
def test_temperature_zero_is_the_argmax_walk(probabilistic):
    gen = torch.Generator().manual_seed(0)
    cand, _, table = make_tables(16, gen)
    seeds = torch.arange(16, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(16 * N, dtype=torch.int64, device=DEVICE).view(16, N) + 5
    tokens, cache, _ = walk(cand, table, torch.zeros(16, device=DEVICE), seeds, pos, probabilistic)
    expected, index = reference_walk(cand.cpu(), table.cpu())
    assert torch.equal(tokens.cpu(), expected)
    assert (index >= KA).any(), "the fixture must exercise the fallback predecessors"
    if probabilistic:  # the cache is written at temperature 0 too: scores on the candidates of the realized predecessor
        for q in range(16):
            prev = 0
            for k in range(N):
                row = torch.full((VOCAB,), float("-inf"))
                row[cand[q, k].cpu()] = table[q, k, prev].cpu()
                assert torch.equal(cache[q, k].cpu(), row)
                prev = int(index[q, k])


def test_temperature_zero_requests_are_unchanged_next_to_sampled_ones():
    gen = torch.Generator().manual_seed(1)
    cand, _, table = make_tables(24, gen)
    seeds = torch.arange(24, dtype=torch.int64, device=DEVICE) + 100
    pos = torch.arange(24 * N, dtype=torch.int64, device=DEVICE).view(24, N) + 5
    temperature = torch.tensor([0.0, 1.0] * 12, device=DEVICE)
    greedy, _, _ = walk(cand, table, torch.zeros(24, device=DEVICE), seeds, pos, False)
    mixed, _, _ = walk(cand, table, temperature, seeds, pos, True)
    assert torch.equal(mixed[0::2], greedy[0::2])
    assert not torch.equal(mixed[1::2], greedy[1::2])  # (the sampled ones really do differ)


# ---------------------------------------------------------------------------------------------------------------------------
# temperature > 0: the sampled token follows its realized predecessor's row, and the cache says so
# ---------------------------------------------------------------------------------------------------------------------------
def chi_square(observed: torch.Tensor, probs: torch.Tensor, min_expected: float = 5.0):
    """Pearson statistic of ``observed`` counts against ``probs`` (small bins merged); returns (chi2, df)."""
    expected = probs * observed.sum()
    big = expected >= min_expected
    obs = torch.cat([observed[big], observed[~big].sum()[None]])
    exp = torch.cat([expected[big], expected[~big].sum()[None]])
    if exp[-1] < min_expected:
        obs, exp = obs[:-1], exp[:-1]
    return float(((obs - exp) ** 2 / exp).sum()), len(obs) - 1


def assert_fits(observed, probs, what, sigmas=7.0):
    chi2, df = chi_square(observed.double().cpu(), probs.double().cpu())
    limit = df + sigmas * math.sqrt(2 * df)
    assert chi2 < limit, f"{what}: chi2 {chi2:.1f} over df {df} (limit {limit:.1f})"


def softmax_on_candidates(row, cand_row, temperature):
    probs = torch.zeros(VOCAB, dtype=torch.float64)
    probs[cand_row.cpu()] = torch.softmax(row.cpu().double() / temperature, dim=0)
    return probs


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_sampled_proposal_is_the_realized_predecessors_row(temperature):
    gen = torch.Generator().manual_seed(2)
    cand1, _, table1 = make_tables(1, gen)
    trials = TRIALS
    cand = cand1.expand(trials, -1, -1).contiguous()
    table = table1.expand(trials, -1, -1, -1).contiguous()
    seeds = torch.arange(trials, dtype=torch.int64, device=DEVICE) * 7919 + 13
    pos = (torch.arange(N, dtype=torch.int64, device=DEVICE) + 9).expand(trials, -1).contiguous()
    tokens, cache, _ = walk(cand, table, torch.full((trials,), temperature, device=DEVICE), seeds, pos, True)

    # candidate index of every drafted token
    index = (cand == tokens[:, :, None]).int().argmax(dim=-1)
    # row 0: the anchor-conditioned row
    counts = torch.bincount(tokens[:, 0].cpu(), minlength=VOCAB)
    assert_fits(counts, softmax_on_candidates(table1[0, 0, 0], cand1[0, 0], temperature), "row 0")
    # row 1, 2: given the index of the predecessor (both conditional rows and fallback rows occur)
    seen_fallback = False
    for k in (1, 2):
        for i in range(K):
            sel = index[:, k - 1] == i
            if int(sel.sum()) < max(60, trials // 15):
                continue
            seen_fallback |= i >= KA
            counts = torch.bincount(tokens[sel, k].cpu(), minlength=VOCAB)
            expected = softmax_on_candidates(table1[0, k, i], cand1[0, k], temperature)
            assert_fits(counts, expected, f"row {k} given predecessor index {i} ({'fallback' if i >= KA else 'conditional'})")
    assert seen_fallback

    # the cache holds, per request and row, the pre-temperature score row of the realized predecessor, -inf elsewhere
    prev = torch.cat([torch.zeros(trials, 1, dtype=torch.int64, device=DEVICE), index[:, :-1]], dim=1)
    realized = table[torch.arange(trials, device=DEVICE)[:, None], torch.arange(N, device=DEVICE)[None], prev]  # [trials, N, K]
    expected_cache = torch.full((trials, N, VOCAB), float("-inf"), device=DEVICE)
    expected_cache.scatter_(2, cand, realized)
    assert torch.equal(cache, expected_cache)


# ---------------------------------------------------------------------------------------------------------------------------
# the whole round: walk + cache + rejection sampler emit the target distribution
# ---------------------------------------------------------------------------------------------------------------------------
def run_rounds(num_tokens, trials, temperature, probabilistic, gen, corrupt_cache=None):
    """Emit ``num_tokens`` tokens for ``trials`` requests from a Markov target (``p(next | prev) = softmax(M[prev] / T)``) with the
    CRL-shaped drafter, round by round (the drafter's tables depend on the anchor token), and return the emitted tokens
    ``[trials, num_tokens]``. ``corrupt_cache(cache, table, index)`` may replace the cached proposal (to see the check fail)."""
    target = (torch.randn(VOCAB, VOCAB, generator=gen) * 1.5).to(DEVICE)
    cand_all, _, table_all = make_tables(VOCAB, gen)
    anchor0 = 3
    out = torch.full((trials, num_tokens), -1, dtype=torch.int64, device=DEVICE)
    count = torch.zeros(trials, dtype=torch.int64, device=DEVICE)
    anchor = torch.full((trials,), anchor0, dtype=torch.int64, device=DEVICE)
    seeds_all = torch.arange(trials, dtype=torch.int64, device=DEVICE) * 104729 + 17
    for _ in range(num_tokens):
        active = (count < num_tokens).nonzero().flatten()
        if active.numel() == 0:
            break
        r = active.numel()
        cand, table = cand_all[anchor[active]], table_all[anchor[active]]
        seeds = seeds_all[active]
        m = count[active]  # tokens emitted so far = position of the anchor token
        sample_pos = m[:, None] + torch.arange(1, N + 1, device=DEVICE)[None]  # position of every drafted token
        temp = torch.full((r,), temperature, device=DEVICE)
        tokens, cache, _ = walk(cand, table, temp, seeds, sample_pos, probabilistic)
        if cache is not None and corrupt_cache is not None:
            index = (cand == tokens[:, :, None]).int().argmax(dim=-1)
            cache = corrupt_cache(cache, cand, table, index)

        draft_sampled = torch.cat([anchor[active][:, None], tokens], dim=1)  # [r, N + 1]: the target's inputs
        target_logits = (target[draft_sampled] / temperature).reshape(r * (N + 1), VOCAB).contiguous()
        sampled, num_sampled = rejection_sample(
            target_logits,
            cache,
            draft_sampled.reshape(-1),
            torch.arange(r + 1, dtype=torch.int32, device=DEVICE) * (N + 1),
            (m[:, None] + torch.arange(N + 1, device=DEVICE)[None]).reshape(-1).to(torch.int32),
            torch.arange(r, dtype=torch.int32, device=DEVICE),
            torch.arange(r, dtype=torch.int32, device=DEVICE).repeat_interleave(N + 1),
            torch.arange(N + 1, dtype=torch.int32, device=DEVICE).repeat(r),
            temp,
            seeds,
            N,
        )
        for j in range(N + 1):  # append what was emitted this round (at most num_tokens in all)
            put = (j < num_sampled) & (count[active] + j < num_tokens)
            rows = active[put]
            out[rows, count[rows] + j] = sampled[put, j]
        count[active] += num_sampled
        anchor[active] = sampled[torch.arange(r), num_sampled - 1]
    return out, target.cpu()


def joint_chi_square(out, target, temperature, anchor0=3):
    p1 = torch.softmax(target[anchor0].double() / temperature, dim=0)
    p2 = torch.softmax(target.double() / temperature, dim=1)  # p2[a, b] = p(t2 = b | t1 = a)
    probs = (p1[:, None] * p2).flatten()
    observed = torch.bincount((out[:, 0] * VOCAB + out[:, 1]).cpu(), minlength=VOCAB * VOCAB)
    return chi_square(observed.double(), probs)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("probabilistic", [False, True], ids=["greedy-draft", "probabilistic-draft"])
def test_round_emits_the_target_distribution(temperature, probabilistic):
    out, target = run_rounds(2, TRIALS, temperature, probabilistic, torch.Generator().manual_seed(3))
    assert (out >= 0).all()
    chi2, df = joint_chi_square(out, target, temperature)
    limit = df + 7 * math.sqrt(2 * df)
    assert chi2 < limit, f"first two tokens: chi2 {chi2:.1f} over df {df} (limit {limit:.1f})"


def test_a_cache_that_is_not_the_sampled_proposal_is_detected():
    """Negative control: a cache sharper than the distribution the walk sampled is a lossy proposal, and the check sees it.

    (A cache holding the wrong predecessor's row is a much smaller distortion -- z of about 2 at 8000 requests on the joint of two
    tokens, below what this statistic resolves -- and is caught instead by the exact comparison in
    ``test_sampled_proposal_is_the_realized_predecessors_row``.)
    """
    temperature = 1.0
    out, target = run_rounds(
        2, TRIALS, temperature, True, torch.Generator().manual_seed(3), corrupt_cache=lambda cache, cand, table, index: cache * 3.0
    )
    chi2, df = joint_chi_square(out, target, temperature)
    assert chi2 > df + 7 * math.sqrt(2 * df), f"a sharpened cache was not detected (chi2 {chi2:.1f}, df {df})"
