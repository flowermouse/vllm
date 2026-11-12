# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused glue of the CRL refinement stage (``CRLSpeculator._score_candidates``).

The stage's GEMMs and its attention are cuBLAS / FlashAttention calls; everything around them was eager PyTorch inside the draft's CUDA graph
(index / cat / expand copies, RMSNorm decomposed into pow / mean / rsqrt / mul, the candidate gather of the LM head, the einsum): about 50 small
kernels, ~450 us of the ~700 us the stage costs per engine step at concurrency 32 (steady-state profile, Progress "并发 32 下 CRL 的优化尝试").
Two Triton kernels and the CUDA norm ops replace them:

* ``_crl_prep_kernel`` (one program per refinement query): gathers the row's last-layer output ``h``, the predecessor's table embedding,
  forms ``r = h + W_e RMSNorm_e(E[a])`` (rounded to bf16 like ``fused_add_rms_norm``), ``x1 = RMSNorm_1(r)`` and the query's rope position.
* ``_crl_score_kernel`` (one program per query): the 16 candidate logits ``x . W[cand]`` straight from the LM head rows (no gathered copy), written
  into the score table.
The norms in between are ``torch.ops._C.rms_norm`` / ``fused_add_rms_norm``.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _crl_prep_kernel(
    hidden_ptr,  # [R, N, 2D] bf16: [normed | pre-norm] rows of the draft; the pre-norm half is read
    hidden_stride,  # row stride (elements) of the [R * N, 2D] view
    anchor_ptr,  # [R] int
    candidate_ptr,  # [R, N, K] int64
    sample_pos_ptr,  # [R * N] int64
    emb_ptr,  # [V, D] bf16: W_e RMSNorm_e(E)
    norm1_ptr,  # [D]
    x1_ptr,  # [R * Q, D] out
    r_ptr,  # [R * Q, D] out (residual stream)
    pos_ptr,  # [R * Q] out int64
    eps,
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
    ka: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0)
    num_q: tl.constexpr = 1 + (num_rows - 1) * ka
    req = p // num_q
    q = p % num_q
    k = tl.where(q == 0, 0, 1 + (q - 1) // ka)  # the draft row this query refines
    j = tl.where(q == 0, 0, (q - 1) % ka)  # rank of its predecessor among the previous row's candidates
    anchor = tl.load(anchor_ptr + req).to(tl.int64)
    previous = tl.load(candidate_ptr + (req * num_rows + tl.maximum(k - 1, 0)) * top_k + j)
    pred = tl.where(q == 0, anchor, previous)

    offs = tl.arange(0, BLOCK)
    m = offs < D
    h = tl.load(hidden_ptr + (req * num_rows + k).to(tl.int64) * hidden_stride + D + offs, mask=m, other=0.0)
    e = tl.load(emb_ptr + pred * D + offs, mask=m, other=0.0)
    r = (h.to(tl.float32) + e.to(tl.float32)).to(tl.bfloat16)  # (the CUDA op adds in the model dtype)
    rf = r.to(tl.float32)
    inv = tl.rsqrt(tl.sum(rf * rf, axis=0) / D + eps)
    w = tl.load(norm1_ptr + offs, mask=m, other=0.0).to(tl.float32)
    x1 = (rf * inv).to(tl.bfloat16).to(tl.float32) * w
    tl.store(r_ptr + p.to(tl.int64) * D + offs, r, mask=m)
    tl.store(x1_ptr + p.to(tl.int64) * D + offs, x1.to(tl.bfloat16), mask=m)
    tl.store(pos_ptr + p, tl.load(sample_pos_ptr + req * num_rows + k))


@triton.jit
def _crl_score_kernel(
    x_ptr,  # [R * Q, D] bf16: the refinement queries after the final norm
    candidate_ptr,  # [R, N, K] int64
    lm_ptr,  # [V, D] bf16
    table_ptr,  # [R, N, K, K] fp32, pre-filled with the rows' own logits
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
    ka: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    num_q: tl.constexpr = 1 + (num_rows - 1) * ka
    req = p // num_q
    q = p % num_q
    k = tl.where(q == 0, 0, 1 + (q - 1) // ka)
    j = tl.where(q == 0, 0, (q - 1) % ka)
    b = tl.arange(0, top_k)
    cand = tl.load(candidate_ptr + (req * num_rows + k) * top_k + b)
    acc = tl.zeros((top_k,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        xv = tl.load(x_ptr + p.to(tl.int64) * D + d, mask=dm, other=0.0).to(tl.float32)
        w = tl.load(lm_ptr + cand[:, None] * D + d[None, :], mask=dm[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
    base = ((req * num_rows + k) * top_k) * top_k
    if q == 0:  # row 1's only predecessor is the anchor: every predecessor slot of the row carries the same scores
        for i in tl.static_range(top_k):
            tl.store(table_ptr + base + i * top_k + b, acc)
    else:
        tl.store(table_ptr + base + j * top_k + b, acc)
