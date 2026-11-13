# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-row confidence of CRL dynamic verification (``crl/fused.py::_crl_confidence_kernel``) against its definition.

Runs on a GPU, or on a CPU through the Triton interpreter (run this file on its own: the interpreter must be switched on before the kernels
are imported). For every sample row the kernel reads the walk's realized score row and the drafted predecessor, never the token of the row:

* MODE_TEMPERATURE: ``max softmax(S / tau)``, tau by the request's temperature class (greedy / sampled);
* MODE_NULL: ``(1 - sigmoid(delta + beta_k)) * max softmax(S)``, ``delta`` of the query of the drafted predecessor (row 1: the anchor's; a
  predecessor beyond the first K_a candidates has no query: the softmax alone);
* padding rows (request state -1) are not written; the confidence buffer is indexed by batch position, the temperature by request state.
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")

if not torch.cuda.is_available():
    import triton
    import triton.language as tl

    import vllm.triton_utils as _triton_utils

    _triton_utils.triton, _triton_utils.tl, _triton_utils.HAS_TRITON = (triton, tl, True)

from vllm.v1.worker.gpu.spec_decode.crl.fused import MODE_NULL, MODE_TEMPERATURE, _crl_confidence_kernel

if not torch.cuda.is_available():
    from triton.runtime.interpreter import InterpretedFunction

    if not isinstance(_crl_confidence_kernel, InterpretedFunction):
        pytest.skip("the Triton kernels were imported before the interpreter was switched on: run this file on its own", allow_module_level=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N, K, KA, R, V = 7, 16, 3, 5, 1000


def _confidence_reference(scores, cand, draft, delta, beta, temperature, tau_greedy, tau_sampled, mode):
    conf = torch.zeros(R, N)
    for req in range(R):
        for k in range(N):
            s = scores[req, k].double()
            tau = (tau_sampled if temperature[req] > 0 else tau_greedy) if mode == MODE_TEMPERATURE else 1.0
            q = torch.softmax(s / tau, 0).max()
            if mode == MODE_NULL:
                if k == 0:
                    slot = 0
                else:
                    hit = (cand[req, k - 1] == draft[req, k - 1]).nonzero()
                    slot = int(hit[0]) if len(hit) else None
                if slot is not None and slot < KA:
                    q = q * (1 - torch.sigmoid(delta[req, k, slot].double() + beta[k].double()))
            conf[req, k] = q.float()
    return conf


@pytest.mark.parametrize("mode", [MODE_TEMPERATURE, MODE_NULL])
def test_confidence_matches_the_definition(mode):
    g = torch.Generator(device=DEVICE).manual_seed(1)
    scores = 3 * torch.randn(R, N, K, generator=g, device=DEVICE)
    cand = torch.stack([torch.randperm(V, generator=g, device=DEVICE)[:K] for _ in range(R * N)]).view(R, N, K)
    # the drafted tokens: usually among the first KA candidates of their row, sometimes further down the list
    pick = torch.randint(0, K, (R, N), generator=g, device=DEVICE)
    pick[0] = torch.tensor([0, 1, 2, KA, K - 1, 0, 1], device=DEVICE)  # includes slots at and beyond K_a
    draft = torch.gather(cand, 2, pick[..., None]).squeeze(-1)
    delta = torch.randn(R, N, KA, generator=g, device=DEVICE)
    beta = torch.randn(N, generator=g, device=DEVICE)
    temperature = torch.tensor([0.0, 1.0, 0.0, 0.7, 0.0], device=DEVICE)
    max_reqs = R + 3  # states are not batch positions
    state = torch.tensor([4, 2, 7, 0, 5], device=DEVICE)
    temp_by_state = torch.zeros(max_reqs, device=DEVICE)
    temp_by_state[state] = temperature
    req_state = state.repeat_interleave(N).to(torch.int32)
    req_state[(R - 1) * N + 3 :] = -1  # padding rows are left untouched
    conf = torch.full((max_reqs, N), -5.0, device=DEVICE)
    _crl_confidence_kernel[(R * N,)](
        scores, cand, draft, delta, beta, req_state, temp_by_state, conf, conf.stride(0), 1.3, 0.8,
        num_rows=N, top_k=K, ka=KA, MODE=mode, num_warps=1,
    )
    ref = _confidence_reference(scores.cpu(), cand.cpu(), draft.cpu(), delta.cpu(), beta.cpu(), temperature.cpu(), 1.3, 0.8, mode)
    got = conf[:R].cpu()  # (batch row r is written at row r: the confidence buffer is indexed by batch position)
    mask = torch.ones(R, N, dtype=torch.bool)
    mask[R - 1, 3:] = False
    torch.testing.assert_close(got[mask], ref[mask], atol=2e-6, rtol=1e-5)
    assert (got[~mask] == -5.0).all()
    assert (got[mask] <= 1.0).all() and (got[mask] > 0).all()
