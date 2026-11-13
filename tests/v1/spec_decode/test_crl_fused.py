# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused glue of the CRL refinement stage (``crl/fused.py``) against its PyTorch definition (GPU test).

``_crl_prep_kernel`` builds, for every refinement query (row 1 once with the anchor token as predecessor, every later row once per
predecessor ``j < crl_pred_topk`` of the row before): ``r = bf16(h + E[pred])``, ``x1 = RMSNorm_1(r)`` and the rope position.
``_crl_score_kernel`` writes the 16 candidate logits ``x . W[cand]`` of every query into the score table (row 1: into all K predecessor slots)
and, with the Null Candidate probe, ``x . w_null`` of every query into the delta table.
(``_crl_confidence_kernel`` is tested in ``test_crl_confidence.py``, which also runs on CPU.)
Run on its own: ``python -m pytest tests/v1/spec_decode/test_crl_fused.py`` (needs CUDA).
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.crl.fused import _crl_prep_kernel, _crl_score_kernel

DEVICE = torch.device("cuda")
N, K, KA, R, V = 7, 16, 3, 5, 1000


@pytest.mark.parametrize("d", [256, 320])  # 320: not a power of two (the kernels mask the tail)
def test_prep_and_score_match_the_pytorch_definition(d):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    hidden = torch.randn(R, N, 2 * d, generator=g, device=DEVICE).bfloat16()
    emb = torch.randn(V, d, generator=g, device=DEVICE).bfloat16()
    norm1 = (torch.rand(d, generator=g, device=DEVICE) + 0.5).bfloat16()
    lm = torch.randn(V, d, generator=g, device=DEVICE).bfloat16()
    anchor_buf = torch.randint(0, V, (R * 9,), generator=g, device=DEVICE)  # the real anchors are a strided view of the input ids
    anchor = anchor_buf[: R * 9 : 9].contiguous()
    cand = torch.stack([torch.randperm(V, generator=g, device=DEVICE)[:K] for _ in range(R * N)]).view(R, N, K)
    sample_pos = torch.arange(R * N, device=DEVICE, dtype=torch.int64) + 100
    eps = 1e-6
    nq = 1 + (N - 1) * KA
    total = R * nq
    hid = hidden.reshape(R * N, 2 * d)

    x1 = torch.empty(total, d, dtype=torch.bfloat16, device=DEVICE)
    r = torch.empty_like(x1)
    pos = torch.empty(total, dtype=torch.int64, device=DEVICE)
    _crl_prep_kernel[(total,)](
        hid, hid.stride(0), anchor, cand, sample_pos, emb, norm1, x1, r, pos, eps,
        num_rows=N, top_k=K, ka=KA, D=d, BLOCK=triton.next_power_of_2(d),
    )

    # the definition: query q of request req
    rows = [0] + [k for k in range(1, N) for _ in range(KA)]
    preds_j = [0] + [j for _ in range(1, N) for j in range(KA)]
    ref_r, ref_x1, ref_pos = [], [], []
    for req in range(R):
        for q in range(nq):
            k, j = rows[q], preds_j[q]
            pred = anchor[req] if q == 0 else cand[req, k - 1, j]
            rq = (hidden[req, k, d:] + emb[pred]).bfloat16()
            rf = rq.float()
            x = (rf * torch.rsqrt(rf.pow(2).mean() + eps)).bfloat16() * norm1
            ref_r.append(rq), ref_x1.append(x), ref_pos.append(sample_pos[req * N + k])
    torch.testing.assert_close(r, torch.stack(ref_r), atol=0, rtol=0)  # the residual stream is exactly bf16(h + E[pred])
    torch.testing.assert_close(x1.float(), torch.stack(ref_x1).float(), atol=3e-2, rtol=2e-2)
    assert torch.equal(pos, torch.stack(ref_pos))

    # scores: x is any [total, d] activation here
    x = torch.randn(total, d, generator=g, device=DEVICE).bfloat16()
    table = torch.full((R, N, K, K), -123.0, device=DEVICE)
    null_w = torch.randn(d, generator=g, device=DEVICE)
    delta = torch.full((R, N, KA), -7.0, device=DEVICE)
    _crl_score_kernel[(total,)](
        x, cand, lm, table, null_w, delta, num_rows=N, top_k=K, ka=KA, D=d, BLOCK_D=min(1024, triton.next_power_of_2(d)), NULL=True
    )
    expected = torch.full_like(table, -123.0)
    expected_delta = torch.full_like(delta, -7.0)
    for req in range(R):
        for q in range(nq):
            k, j = rows[q], preds_j[q]
            s = (lm[cand[req, k]].float() @ x[req * nq + q].float())  # [K]
            expected_delta[req, k, j] = null_w @ x[req * nq + q].float()
            if q == 0:
                expected[req, 0, :, :] = s  # row 1: every predecessor slot carries the same scores
            else:
                expected[req, k, j, :] = s
    torch.testing.assert_close(table, expected, atol=2e-2, rtol=2e-3)
    torch.testing.assert_close(delta, expected_delta, atol=5e-2, rtol=2e-3)

