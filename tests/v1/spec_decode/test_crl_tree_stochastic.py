# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rigorous statistical & edge-case tests for CRL stochastic without-replacement
tree drafting and recursive residual tree rejection sampling.
"""

import math

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.crl.tree import CRLTreeState


def _chi2_gof_pvalue(obs: np.ndarray, exp: np.ndarray) -> float:
    """Chi-square goodness-of-fit p-value (combining tiny bins with exp < 5)."""
    mask = exp >= 5.0
    o = obs[mask].astype(np.float64)
    e = exp[mask].astype(np.float64)
    if (~mask).any():
        o_tail = float(obs[~mask].sum())
        e_tail = float(exp[~mask].sum())
        if e_tail >= 5.0:
            o = np.append(o, o_tail)
            e = np.append(e, e_tail)
    stat = float(np.sum((o - e) ** 2 / e))
    df = max(1, len(o) - 1)
    # Wilson-Hilferty normal approximation to upper tail of chi-square(df)
    # (exact to <1e-3 for df >= 10; use scipy if installed).
    try:
        from scipy.stats import chi2

        return float(chi2.sf(stat, df))
    except ImportError:
        z = ((stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(
            2.0 / (9.0 * df)
        )
        return 0.5 * math.erfc(z / math.sqrt(2.0))


def _sample_single_step_gpu(
    target_logits_1d: torch.Tensor,
    draft_scores_1d: torch.Tensor,
    cand_ids_1d: torch.Tensor,
    budget: int,
    max_branch: int,
    n_samples: int,
    temperature: float = 1.0,
    batch_size: int = 256,
) -> tuple[np.ndarray, float]:
    """Run real Triton build_trees + verify_trees_stochastic for n_samples draws."""
    device = target_logits_1d.device
    V = target_logits_1d.shape[0]
    K = cand_ids_1d.shape[0]
    N_ROWS = 4
    state = CRLTreeState(
        max_num_reqs=batch_size,
        budget=budget,
        draft_rows=N_ROWS,
        top_k=K,
        device=device,
        max_branch=max_branch,
    )
    state.stochastic_enabled = True

    cand_ids = cand_ids_1d.view(1, 1, K).expand(batch_size, N_ROWS, K).contiguous()
    scores = draft_scores_1d.view(1, 1, 1, K).expand(batch_size, N_ROWS, K, K).contiguous()
    idx_map = torch.arange(batch_size, dtype=torch.int64, device=device)
    sample_idx_map = idx_map.repeat_interleave(N_ROWS)
    out_draft = torch.zeros(batch_size, budget, dtype=torch.int64, device=device)
    temps = torch.full((batch_size,), temperature, dtype=torch.float32, device=device)
    cu_logits = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * (budget + 1)
    logits = (target_logits_1d / temperature).view(1, V).expand(batch_size * (budget + 1), V).contiguous()
    positions = torch.arange(batch_size * (budget + 1), dtype=torch.int64, device=device) % (budget + 1)

    counts = np.zeros(V, dtype=np.int64)
    accepted_first = 0
    total = 0
    rng = np.random.default_rng(12345)

    n_batches = (n_samples + batch_size - 1) // batch_size
    for b in range(n_batches):
        cur = min(batch_size, n_samples - total)
        seeds = torch.from_numpy(
            rng.integers(-(1 << 62), 1 << 62, size=batch_size, dtype=np.int64)
        ).to(device)
        sample_pos = torch.full(
            (batch_size * N_ROWS,), 10 + b, dtype=torch.int64, device=device
        )
        state.build_trees(
            cand_ids,
            scores,
            batch_size,
            sample_idx_map,
            out_draft,
            pad_token=0,
            temperature=temps,
            seeds=seeds,
            sample_positions=sample_pos,
        )
        sampled, acc_nodes, num_sampled, _ = state.verify_trees_stochastic(
            logits,
            idx_map,
            cu_logits,
            positions,
            seeds,
            temps,
            batch_size,
        )
        tok0 = sampled[:cur, 0].cpu().numpy()
        ns = num_sampled[:cur].cpu().numpy()
        np.add.at(counts, tok0, 1)
        accepted_first += int((ns >= 2).sum())
        total += cur

    return counts, accepted_first / total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("budget", [1, 15, 31])
@pytest.mark.parametrize("regime", ["high_entropy", "low_entropy", "top_p_0.9"])
def test_single_step_distribution_lossless(budget: int, regime: str):
    """Test 1: Empirical first-token distribution matches exact target p (chi2 p > 0.01, TV < 0.005)."""
    device = torch.device("cuda")
    V, K = 64, 16
    g = torch.Generator(device=device).manual_seed(42)
    if regime == "high_entropy":
        t_logits = torch.randn(V, device=device, generator=g) * 0.8
    elif regime == "low_entropy":
        t_logits = torch.randn(V, device=device, generator=g) * 2.5
    else:
        t_logits = torch.randn(V, device=device, generator=g) * 1.2
        probs_raw = torch.softmax(t_logits, dim=-1)
        s_probs, s_idx = torch.sort(probs_raw, descending=True)
        cdf = torch.cumsum(s_probs, dim=-1)
        drop = (cdf - s_probs) > 0.9
        t_logits = t_logits.clone()
        t_logits[s_idx[drop]] = float("-inf")

    p_true = torch.softmax(t_logits, dim=-1).cpu().numpy()
    # Mismatched draft distribution over K=16 candidates
    cand_ids = torch.arange(K, dtype=torch.int64, device=device)
    d_scores = t_logits[:K].clone()
    d_scores = torch.where(torch.isfinite(d_scores), d_scores + torch.randn(K, device=device, generator=g) * 0.7, torch.tensor(-5.0, device=device))

    N = 400_000
    counts, _ = _sample_single_step_gpu(
        t_logits, d_scores, cand_ids, budget=budget, max_branch=4, n_samples=N
    )
    emp = counts / N
    tv = 0.5 * float(np.abs(emp - p_true).sum())
    pval = _chi2_gof_pvalue(counts, p_true * N)
    assert tv < 0.005, f"TV={tv:.5f} too high (budget={budget}, regime={regime})"
    assert pval > 0.01, f"chi2 p={pval:.5f} failed (budget={budget}, regime={regime})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multistep_joint_distribution_lossless():
    """Test 2: Depth-2 joint distribution P(x1, x2) matches autoregressive target."""
    device = torch.device("cuda")
    V, K = 8, 8
    N_ROWS = 2
    budget = 15
    batch_size = 256
    N = 400_000

    g = torch.Generator(device=device).manual_seed(7)
    # Row 0 target logits p0(x1)
    t0 = torch.randn(V, device=device, generator=g) * 0.9
    p0 = torch.softmax(t0, dim=-1).cpu().numpy()
    # Row 1 conditional target logits p1(x2 | x1)
    t1_table = torch.randn(V, V, device=device, generator=g) * 1.1
    p1_table = torch.softmax(t1_table, dim=-1).cpu().numpy()
    p_joint_true = (p0[:, None] * p1_table).reshape(-1)

    # Mismatched draft scores
    d0 = t0 + torch.randn(K, device=device, generator=g) * 0.6
    d1 = t1_table + torch.randn(K, K, device=device, generator=g) * 0.6

    state = CRLTreeState(
        max_num_reqs=batch_size,
        budget=budget,
        draft_rows=N_ROWS,
        top_k=K,
        device=device,
        max_branch=4,
    )
    state.stochastic_enabled = True

    cand_ids = torch.arange(K, dtype=torch.int64, device=device).view(1, 1, K).expand(batch_size, N_ROWS, K).contiguous()
    scores = torch.zeros(batch_size, N_ROWS, K, K, dtype=torch.float32, device=device)
    scores[:, 0, 0, :] = d0
    scores[:, 1, :, :] = d1
    idx_map = torch.arange(batch_size, dtype=torch.int64, device=device)
    sample_idx_map = idx_map.repeat_interleave(N_ROWS)
    out_draft = torch.zeros(batch_size, budget, dtype=torch.int64, device=device)
    temps = torch.ones(batch_size, dtype=torch.float32, device=device)
    cu_logits = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * (budget + 1)
    positions = torch.arange(batch_size * (budget + 1), dtype=torch.int64, device=device) % (budget + 1)

    joint_counts = np.zeros(V * V, dtype=np.int64)
    rng = np.random.default_rng(999)
    total = 0
    n_batches = (N + batch_size - 1) // batch_size
    for b in range(n_batches):
        cur = min(batch_size, N - total)
        seeds = torch.from_numpy(
            rng.integers(-(1 << 62), 1 << 62, size=batch_size, dtype=np.int64)
        ).to(device)
        sample_pos = torch.full((batch_size * N_ROWS,), 20 + b, dtype=torch.int64, device=device)
        state.build_trees(
            cand_ids, scores, batch_size, sample_idx_map, out_draft,
            pad_token=0, temperature=temps, seeds=seeds, sample_positions=sample_pos,
        )
        # Build exact target logits for each tree node: root uses t0, depth-1 node v uses t1_table[tok(v)]
        logits = torch.zeros(batch_size, budget + 1, V, dtype=torch.float32, device=device)
        logits[:, 0, :] = t0
        depth1_toks = state.tree_tokens[:batch_size, :budget]  # [B, budget]
        logits[:, 1:, :] = t1_table[depth1_toks]
        sampled, _, num_sampled, _ = state.verify_trees_stochastic(
            logits.view(-1, V), idx_map, cu_logits, positions, seeds, temps, batch_size
        )
        s_np = sampled[:cur].cpu().numpy()
        ns_np = num_sampled[:cur].cpu().numpy()
        for i in range(cur):
            x1 = int(s_np[i, 0])
            if ns_np[i] >= 2:
                x2 = int(s_np[i, 1])
            else:
                # If step 1 rejected, draw x2 from exact p1(. | x1) (next step in AR)
                x2 = int(rng.choice(V, p=p1_table[x1]))
            joint_counts[x1 * V + x2] += 1
        total += cur

    emp = joint_counts / N
    tv = 0.5 * float(np.abs(emp - p_joint_true).sum())
    pval = _chi2_gof_pvalue(joint_counts, p_joint_true * N)
    assert tv < 0.005, f"Multi-step TV={tv:.5f} too high"
    assert pval > 0.01, f"Multi-step chi2 p={pval:.5f} failed"


def test_negative_control_broken_residual_fails():
    """Test 3: Removing residual update on reject reliably fails the chi2/TV test."""
    rng = np.random.default_rng(0)
    p = np.array([0.50, 0.25, 0.15, 0.10], dtype=np.float64)
    q = np.array([0.10, 0.40, 0.30, 0.20], dtype=np.float64)
    N = 50_000
    counts = np.zeros(4, dtype=np.int64)
    for _ in range(N):
        x = rng.choice(4, p=q)
        if rng.random() < min(1.0, p[x] / q[x]):
            out = x
        else:
            # Broken: resample from unmodified p instead of residual norm(max(0, p-q))
            out = rng.choice(4, p=p)
        counts[out] += 1
    tv = 0.5 * float(np.abs(counts / N - p).sum())
    pval = _chi2_gof_pvalue(counts, p * N)
    assert tv > 0.03 and pval < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_acceptance_rate_chain_and_dominance():
    """Test 4: m=1 matches theoretical sum(min(p,q)); m=4 strictly beats strict-match."""
    device = torch.device("cuda")
    V, K = 32, 16
    g = torch.Generator(device=device).manual_seed(123)
    t_logits = torch.randn(V, device=device, generator=g) * 1.0
    cand_ids = torch.arange(K, dtype=torch.int64, device=device)
    d_scores = t_logits[:K] + torch.randn(K, device=device, generator=g) * 0.4

    p = torch.softmax(t_logits, dim=-1).cpu().numpy()
    q_full = np.zeros(V, dtype=np.float64)
    q_full[:K] = torch.softmax(d_scores, dim=-1).cpu().numpy()
    theo_chain = float(np.minimum(p, q_full).sum())

    # m = 1 (B=1)
    _, acc_m1 = _sample_single_step_gpu(
        t_logits, d_scores, cand_ids, budget=1, max_branch=1, n_samples=100_000
    )
    assert abs(acc_m1 - theo_chain) < 0.006, f"acc_m1={acc_m1:.4f} vs theo={theo_chain:.4f}"

    # m = 4 (B=4, max_branch=4) vs deterministic strict-match top-4 sum
    top4_idx = np.argsort(-q_full[:K])[:4]
    strict_match_4 = float(p[top4_idx].sum())
    _, acc_m4 = _sample_single_step_gpu(
        t_logits, d_scores, cand_ids, budget=4, max_branch=4, n_samples=100_000
    )
    assert acc_m4 > strict_match_4 + 0.01, (
        f"acc_m4={acc_m4:.4f} should strictly beat strict_match_4={strict_match_4:.4f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_edge_cases_mixed_batch_prefill_and_zero_residual():
    """Test 5: Mixed T=0/T=1 batch, n_logits=1 prefill, degenerate S->0, exact p==q (Z->0)."""
    device = torch.device("cuda")
    V, K, B, N_ROWS = 64, 16, 15, 4
    num_reqs = 4
    state = CRLTreeState(
        max_num_reqs=num_reqs, budget=B, draft_rows=N_ROWS, top_k=K, device=device
    )
    state.stochastic_enabled = True

    cand_ids = torch.arange(K, dtype=torch.int64, device=device).view(1, 1, K).expand(num_reqs, N_ROWS, K).contiguous()
    scores = torch.zeros(num_reqs, N_ROWS, K, K, dtype=torch.float32, device=device)
    # Req 0: T=0 greedy
    # Req 1: T=1 prefill-only (n_logits=1)
    # Req 2: T=1 degenerate 1-hot draft (S->0 after child 0)
    scores[2, :, :, :] = -1e9
    scores[2, :, :, 3] = 0.0
    # Req 3: T=1 exact p == q (Z -> 0 if rejected)
    scores[3, :, :, :] = torch.linspace(2.0, -2.0, K, device=device)

    idx_map = torch.arange(num_reqs, dtype=torch.int64, device=device)
    sample_idx_map = idx_map.repeat_interleave(N_ROWS)
    out_draft = torch.zeros(num_reqs, B, dtype=torch.int64, device=device)
    temps = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    seeds = torch.tensor([11, 22, 33, 44], dtype=torch.int64, device=device)
    sample_pos = torch.full((num_reqs * N_ROWS,), 5, dtype=torch.int64, device=device)

    state.build_trees(
        cand_ids, scores, num_reqs, sample_idx_map, out_draft,
        pad_token=0, temperature=temps, seeds=seeds, sample_positions=sample_pos,
    )

    # cu_num_logits: req 1 has n_logits=1 (prefill), others have B+1
    lens = [B + 1, 1, B + 1, B + 1]
    cu = torch.tensor([0, 16, 17, 33, 49], dtype=torch.int32, device=device)
    logits = torch.full((49, V), -1e9, dtype=torch.float32, device=device)
    logits[:16, :K] = 0.0
    logits[:16, 5] = 10.0  # greedy target = 5
    logits[16, :10] = 1.0  # prefill
    logits[17:33, :K] = 0.0
    logits[33:49, :K] = torch.linspace(2.0, -2.0, K, device=device)
    positions = torch.arange(49, dtype=torch.int64, device=device)

    sampled, acc_nodes, num_sampled, num_rejected = state.verify_trees_stochastic(
        logits, idx_map, cu, positions, seeds, temps, num_reqs
    )
    assert num_sampled[0].item() >= 1 and sampled[0, 0].item() == 5
    assert num_sampled[1].item() == 1 and num_rejected[1].item() == 0
    assert 0 <= sampled[1, 0].item() < 10
    assert num_sampled[2].item() >= 1 and num_sampled[3].item() >= 1
    for r in range(num_reqs):
        ns = num_sampled[r].item()
        assert ns + num_rejected[r].item() == lens[r]
        assert (sampled[r, :ns] >= 0).all() and (sampled[r, :ns] < V).all()
