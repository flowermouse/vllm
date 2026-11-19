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

Dynamic verification (``enable_adaptive_verification``) needs one confidence per drafted row, written into the draft graph:

* ``_crl_score_kernel`` can also dot every query with the Null Candidate probe ``w_null`` (a 17th LM-head row, ``NULL=True``) into ``delta_ptr``;
* ``_crl_confidence_kernel`` (one program per row, after the walk) turns the walked path's realized scores into ``draft_token_confidence_probs``:
  ``max softmax(S / tau)`` (``MODE_TEMPERATURE``, tau per request class: greedy / sampled) or ``(1 - sigmoid(delta + beta_k)) * max softmax(S)``
  (``MODE_NULL``). Both read only the score row of the *drafted predecessor* (the walk's previous token), never the token drawn on the row itself,
  so trimming on them leaves rejection sampling lossless (the condition of DSpark Sec. 3.2.2).
"""

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
    j = tl.where(
        q == 0, 0, (q - 1) % ka
    )  # rank of its predecessor among the previous row's candidates
    anchor = tl.load(anchor_ptr + req).to(tl.int64)
    previous = tl.load(
        candidate_ptr + (req * num_rows + tl.maximum(k - 1, 0)) * top_k + j
    )
    pred = tl.where(q == 0, anchor, previous)

    offs = tl.arange(0, BLOCK)
    m = offs < D
    h = tl.load(
        hidden_ptr + (req * num_rows + k).to(tl.int64) * hidden_stride + D + offs,
        mask=m,
        other=0.0,
    )
    e = tl.load(emb_ptr + pred * D + offs, mask=m, other=0.0)
    r = (h.to(tl.float32) + e.to(tl.float32)).to(
        tl.bfloat16
    )  # (the CUDA op adds in the model dtype)
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
    null_w_ptr,  # [D] the Null Candidate probe (read only when NULL)
    delta_ptr,  # [R, N, ka] fp32 out: x . w_null of the query (predecessor slot j of row k; read only when NULL)
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
    ka: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NULL: tl.constexpr,
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
    dacc = tl.zeros((1,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        xv = tl.load(x_ptr + p.to(tl.int64) * D + d, mask=dm, other=0.0).to(tl.float32)
        w = tl.load(
            lm_ptr + cand[:, None] * D + d[None, :], mask=dm[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
        if NULL:
            wn = tl.load(null_w_ptr + d, mask=dm, other=0.0).to(tl.float32)
            dacc += tl.sum(wn * xv, axis=0)
    if NULL:
        tl.store(delta_ptr + (req * num_rows + k) * ka + j + tl.arange(0, 1), dacc)
    base = ((req * num_rows + k) * top_k) * top_k
    if (
        q == 0
    ):  # row 1's only predecessor is the anchor: every predecessor slot of the row carries the same scores
        for i in tl.static_range(top_k):
            tl.store(table_ptr + base + i * top_k + b, acc)
    else:
        tl.store(table_ptr + base + j * top_k + b, acc)


MODE_TEMPERATURE = 0
MODE_NULL = 1


@triton.jit
def _crl_confidence_kernel(
    scores_ptr,  # [R, N, K] fp32: the walk's realized score rows (the row of the drafted predecessor; raw, not divided by the temperature)
    candidate_ptr,  # [R, N, K] int64
    draft_ptr,  # [R, N] int64: the drafted tokens
    delta_ptr,  # [R, N, ka] fp32: x . w_null per query (MODE_NULL)
    beta_ptr,  # [N] fp32: the Null Candidate's per-row bias (MODE_NULL)
    req_state_ptr,  # [R * N] int: request-state index of every sample row, -1 for graph padding
    temperature_ptr,  # [max_num_reqs] fp32: the request's sampling temperature
    conf_ptr,  # [max_num_reqs, N] fp32 out
    conf_stride,
    tau_greedy,
    tau_sampled,
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
    ka: tl.constexpr,
    MODE: tl.constexpr,
):
    p = tl.program_id(0)
    req = p // num_rows
    k = p % num_rows
    state = tl.load(req_state_ptr + p)
    if state < 0:
        return
    b = tl.arange(0, top_k)
    s = tl.load(scores_ptr + p * top_k + b)
    if MODE == 0:
        temperature = tl.load(temperature_ptr + state)
        tau = tl.where(temperature > 0.0, tau_sampled, tau_greedy)
    else:
        tau = 1.0
    z = (s - tl.max(s, axis=0)) / tau
    conf = 1.0 / tl.sum(tl.exp(z), axis=0)  # the largest softmax probability of the row
    if MODE == 1:
        # the query of the drafted predecessor: row 0 has the anchor only; row k > 0 has one per predecessor among the previous row's first ka
        # candidates. A predecessor further down the list has no query (the row's own logits were used): no delta, the softmax alone.
        prev_tok = tl.load(draft_ptr + req * num_rows + tl.maximum(k - 1, 0))
        prev_cand = tl.load(
            candidate_ptr + (req * num_rows + tl.maximum(k - 1, 0)) * top_k + b
        )
        hit = prev_cand == prev_tok
        idx = tl.sum(tl.where(hit, b, 0), axis=0)
        found = tl.sum(hit.to(tl.int32), axis=0) > 0
        has_query = (k == 0) | (found & (idx < ka))
        slot = tl.where(k == 0, 0, idx)
        slot = tl.minimum(slot, ka - 1)
        delta = tl.load(delta_ptr + (req * num_rows + k) * ka + slot) + tl.load(
            beta_ptr + k
        )
        recall = 1.0 - 1.0 / (1.0 + tl.exp(-delta))
        conf = tl.where(has_query, conf * recall, conf)
    tl.store(conf_ptr + req * conf_stride + k, conf)


@triton.jit
def _calib_store_kernel(
    scores_ptr,  # [R, N, K] fp32: the walk's realized score rows (batch position)
    req_state_ptr,  # [R * N] int: request-state index of every sample row, -1 for graph padding
    out_ptr,  # [max_num_reqs, N, K] fp32: the rows of the latest proposal of every request state
    valid_ptr,  # [max_num_reqs] int32: 1 = the state's rows are those of a proposal that has not been verified yet
    num_rows: tl.constexpr,
    top_k: tl.constexpr,
):
    p = tl.program_id(0)
    state = tl.load(req_state_ptr + p)
    if state < 0:
        return
    b = tl.arange(0, top_k)
    tl.store(
        out_ptr + state.to(tl.int64) * (num_rows * top_k) + (p % num_rows) * top_k + b,
        tl.load(scores_ptr + p * top_k + b),
    )
    tl.store(valid_ptr + state, 1)


@triton.jit
def _crl_prep_varlen_kernel(
    hidden_ptr,  # packed drafted rows [sum(d), 2D]; the pre-norm half starts at D
    hidden_stride,
    anchor_ptr,  # [R]
    candidate_ptr,  # [R, STEPS, K]
    sample_pos_ptr,  # [R, STEPS]
    emb_ptr,
    norm1_ptr,
    x1_ptr,
    r_ptr,
    pos_ptr,
    req_ptr,  # [Q] which request this packed query belongs to
    q_ptr,  # [Q] query index inside that request
    row_cu_ptr,  # [R] first packed hidden row of the request
    eps,
    STEPS: tl.constexpr,
    top_k: tl.constexpr,
    ka: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per refinement query. Depth differs per request, so the query is addressed by ``req_ptr`` / ``q_ptr`` instead of a uniform stride."""
    p = tl.program_id(0)
    req = tl.load(req_ptr + p)
    # Static-shape graph padding. Write zeros so the captured FA kernel does
    # not read uninitialized x1, then leave.
    if req < 0:
        offs = tl.arange(0, BLOCK)
        m = offs < D
        z = tl.zeros((BLOCK,), dtype=tl.float32).to(tl.bfloat16)
        tl.store(r_ptr + p.to(tl.int64) * D + offs, z, mask=m)
        tl.store(x1_ptr + p.to(tl.int64) * D + offs, z, mask=m)
        tl.store(pos_ptr + p, 0)
        return
    q = tl.load(q_ptr + p)
    k = tl.where(q == 0, 0, 1 + (q - 1) // ka)
    j = tl.where(q == 0, 0, (q - 1) % ka)
    anchor = tl.load(anchor_ptr + req).to(tl.int64)
    previous = tl.load(candidate_ptr + (req * STEPS + tl.maximum(k - 1, 0)) * top_k + j)
    pred = tl.where(q == 0, anchor, previous)
    row = tl.load(row_cu_ptr + req) + k
    offs = tl.arange(0, BLOCK)
    m = offs < D
    h = tl.load(
        hidden_ptr + row.to(tl.int64) * hidden_stride + D + offs, mask=m, other=0.0
    )
    e = tl.load(emb_ptr + pred * D + offs, mask=m, other=0.0)
    r = (h.to(tl.float32) + e.to(tl.float32)).to(tl.bfloat16)
    rf = r.to(tl.float32)
    inv = tl.rsqrt(tl.sum(rf * rf, axis=0) / D + eps)
    w = tl.load(norm1_ptr + offs, mask=m, other=0.0).to(tl.float32)
    x1 = (rf * inv).to(tl.bfloat16).to(tl.float32) * w
    tl.store(r_ptr + p.to(tl.int64) * D + offs, r, mask=m)
    tl.store(x1_ptr + p.to(tl.int64) * D + offs, x1.to(tl.bfloat16), mask=m)
    tl.store(pos_ptr + p, tl.load(sample_pos_ptr + req * STEPS + k))


@triton.jit
def _crl_score_varlen_kernel(
    x_ptr,  # [Q, D]
    candidate_ptr,  # [R, STEPS, K]
    lm_ptr,
    table_ptr,  # [R, STEPS, K, K], pre-filled with the rows' own logits
    null_w_ptr,
    delta_ptr,  # [R, STEPS, ka]. STEPS is the full width (15), not this request's depth.
    req_ptr,
    q_ptr,
    STEPS: tl.constexpr,
    top_k: tl.constexpr,
    ka: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NULL: tl.constexpr,
):
    p = tl.program_id(0)
    req = tl.load(req_ptr + p)
    if req < 0:
        return
    q = tl.load(q_ptr + p)
    k = tl.where(q == 0, 0, 1 + (q - 1) // ka)
    j = tl.where(q == 0, 0, (q - 1) % ka)
    b = tl.arange(0, top_k)
    cand = tl.load(candidate_ptr + (req * STEPS + k) * top_k + b)
    acc = tl.zeros((top_k,), dtype=tl.float32)
    dacc = tl.zeros((1,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dm = d < D
        xv = tl.load(x_ptr + p.to(tl.int64) * D + d, mask=dm, other=0.0).to(tl.float32)
        w = tl.load(
            lm_ptr + cand[:, None] * D + d[None, :], mask=dm[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)
        if NULL:
            wn = tl.load(null_w_ptr + d, mask=dm, other=0.0).to(tl.float32)
            dacc += tl.sum(wn * xv, axis=0)
    if NULL:
        tl.store(delta_ptr + (req * STEPS + k) * ka + j + tl.arange(0, 1), dacc)
    base = ((req * STEPS + k) * top_k) * top_k
    if q == 0:
        for i in tl.static_range(top_k):
            tl.store(table_ptr + base + i * top_k + b, acc)
    else:
        tl.store(table_ptr + base + j * top_k + b, acc)


@triton.jit
def _pava4_kernel(src_ptr, dst_ptr):
    """Decreasing PAVA on each row of ``[R, 4]``. One program per row, no host sync."""
    row = tl.program_id(0)
    b = row * 4
    v0 = tl.load(src_ptr + b + 0).to(tl.float32)
    v1 = tl.load(src_ptr + b + 1).to(tl.float32)
    v2 = tl.load(src_ptr + b + 2).to(tl.float32)
    v3 = tl.load(src_ptr + b + 3).to(tl.float32)
    # Stack of pooled blocks. Push four values, merging ascents.
    s0 = v0
    w0 = 1.0
    n = 1
    s1 = 0.0
    w1 = 0.0
    s2 = 0.0
    w2 = 0.0
    s3 = 0.0
    w3 = 0.0

    s1 = v1
    w1 = 1.0
    n = 2
    if s0 < s1:
        tw = w0 + w1
        m = (s0 * w0 + s1 * w1) / tw
        s0 = m
        w0 = tw
        n = 1

    if n == 1:
        s1 = v2
        w1 = 1.0
        n = 2
        if s0 < s1:
            tw = w0 + w1
            m = (s0 * w0 + s1 * w1) / tw
            s0 = m
            w0 = tw
            n = 1
    else:
        s2 = v2
        w2 = 1.0
        n = 3
        if s1 < s2:
            tw = w1 + w2
            m = (s1 * w1 + s2 * w2) / tw
            s1 = m
            w1 = tw
            n = 2
            if s0 < s1:
                tw = w0 + w1
                m = (s0 * w0 + s1 * w1) / tw
                s0 = m
                w0 = tw
                n = 1

    if n == 1:
        s1 = v3
        w1 = 1.0
        n = 2
        if s0 < s1:
            tw = w0 + w1
            m = (s0 * w0 + s1 * w1) / tw
            s0 = m
            w0 = tw
            n = 1
    elif n == 2:
        s2 = v3
        w2 = 1.0
        n = 3
        if s1 < s2:
            tw = w1 + w2
            m = (s1 * w1 + s2 * w2) / tw
            s1 = m
            w1 = tw
            n = 2
            if s0 < s1:
                tw = w0 + w1
                m = (s0 * w0 + s1 * w1) / tw
                s0 = m
                w0 = tw
                n = 1
    else:
        s3 = v3
        w3 = 1.0
        n = 4
        if s2 < s3:
            tw = w2 + w3
            m = (s2 * w2 + s3 * w3) / tw
            s2 = m
            w2 = tw
            n = 3
            if s1 < s2:
                tw = w1 + w2
                m = (s1 * w1 + s2 * w2) / tw
                s1 = m
                w1 = tw
                n = 2
                if s0 < s1:
                    tw = w0 + w1
                    m = (s0 * w0 + s1 * w1) / tw
                    s0 = m
                    w0 = tw
                    n = 1

    o0 = s0
    o1 = s0
    o2 = s0
    o3 = s0
    if n >= 2:
        if w0 < 1.5:
            o1 = s1
        if w0 < 2.5:
            o2 = s1
        if w0 < 3.5:
            o3 = s1
    if n >= 3:
        cut = w0 + w1
        if cut < 2.5:
            o2 = s2
        if cut < 3.5:
            o3 = s2
    if n >= 4:
        o3 = s3
    tl.store(dst_ptr + b + 0, o0)
    tl.store(dst_ptr + b + 1, o1)
    tl.store(dst_ptr + b + 2, o2)
    tl.store(dst_ptr + b + 3, o3)


@triton.jit
def _elastic_scan_kernel(
    q_ptr,  # [R] int32 query rows
    limits_ptr,  # [R] int32 draft depth
    row_cu_ptr,  # [R] int32 first packed hidden row
    q_cu_ptr,  # [R + 1] int32 CRL query starts, sums to C_PAD
    R: tl.constexpr,
    KA: tl.constexpr,
    C_PAD: tl.constexpr,
    MAX_NQ: tl.constexpr,
):
    """Depths, hidden-row offsets, and per-request CRL spans for one bucket.

    Spare CRL slots are appended to requests that still have room under
    ``MAX_NQ``, so no span exceeds it and the packed total is ``C_PAD``.
    """
    zero = tl.load(q_ptr).to(tl.int32) * 0
    c_tot = zero
    d_tot = zero
    for r in tl.static_range(R):
        q = tl.load(q_ptr + r).to(tl.int32)
        dd = tl.where(q > 0, q - 1, zero)
        nn = tl.where(dd > 0, 1 + (dd - 1) * KA, zero)
        tl.store(limits_ptr + r, dd)
        tl.store(row_cu_ptr + r, d_tot)
        d_tot += dd
        c_tot += nn
    pad = C_PAD - c_tot
    pos = zero
    tl.store(q_cu_ptr, zero)
    for r in tl.static_range(R):
        dd = tl.load(limits_ptr + r).to(tl.int32)
        nn = tl.where(dd > 0, 1 + (dd - 1) * KA, zero)
        room = MAX_NQ - nn
        add = tl.where((room > 0) & (pad > 0), tl.minimum(room, pad), zero)
        pad -= add
        pos += nn + add
        tl.store(q_cu_ptr + r + 1, pos)


@triton.jit
def _elastic_hidden_kernel(
    sample_idx_ptr,  # [R, STEPS] int64
    limits_ptr,
    row_cu_ptr,
    hidden_idx_ptr,  # [D_PAD]
    cand_pos_ptr,  # [D_PAD], -1 = padding
    R: tl.constexpr,
    STEPS: tl.constexpr,
    D_PAD: tl.constexpr,
):
    p = tl.program_id(0)
    if p >= D_PAD:
        return
    for r in tl.static_range(R):
        start = tl.load(row_cu_ptr + r)
        dd = tl.load(limits_ptr + r)
        if (start <= p) & (p < start + dd):
            k = p - start
            tl.store(hidden_idx_ptr + p, tl.load(sample_idx_ptr + r * STEPS + k))
            tl.store(cand_pos_ptr + p, r * STEPS + k)
            return
    tl.store(hidden_idx_ptr + p, 0)
    tl.store(cand_pos_ptr + p, -1)


@triton.jit
def _elastic_crl_kernel(
    limits_ptr,
    q_cu_ptr,  # [R + 1]
    crl_req_ptr,  # [C_PAD], -1 = padding
    crl_qlocal_ptr,
    R: tl.constexpr,
    KA: tl.constexpr,
    C_PAD: tl.constexpr,
):
    p = tl.program_id(0)
    if p >= C_PAD:
        return
    for r in tl.static_range(R):
        start = tl.load(q_cu_ptr + r)
        end = tl.load(q_cu_ptr + r + 1)
        if (start <= p) & (p < end):
            dd = tl.load(limits_ptr + r)
            nn = 0
            if dd > 0:
                nn = 1 + (dd - 1) * KA
            local = p - start
            if local < nn:
                tl.store(crl_req_ptr + p, r)
                tl.store(crl_qlocal_ptr + p, local)
            else:
                tl.store(crl_req_ptr + p, -1)
                tl.store(crl_qlocal_ptr + p, 0)
            return
    tl.store(crl_req_ptr + p, -1)
    tl.store(crl_qlocal_ptr + p, 0)


@triton.jit
def _elastic_scatter_kernel(
    ids_ptr,  # [D_PAD, K] int64
    logits_ptr,  # [D_PAD, K] fp32
    pos_ptr,  # [D_PAD] int32
    cand_ptr,  # [R * STEPS, K] int64
    unary_ptr,  # [R * STEPS, K] fp32
    K: tl.constexpr,
):
    p = tl.program_id(0)
    dst = tl.load(pos_ptr + p)
    if dst < 0:
        return
    b = tl.arange(0, K)
    tl.store(
        cand_ptr + dst.to(tl.int64) * K + b, tl.load(ids_ptr + p.to(tl.int64) * K + b)
    )
    tl.store(
        unary_ptr + dst.to(tl.int64) * K + b,
        tl.load(logits_ptr + p.to(tl.int64) * K + b),
    )


@triton.jit
def _elastic_fix_qsl_kernel(
    qsl_ptr,  # [R + 1] int32, prepare wrote the real starts
    seq_ptr,  # [R] int32
    q_cu_ptr,  # [R + 1] real query rows, tail holds Q_tot
    live_ptr,  # [1] int32
    input_ids_ptr,  # [max_num_tokens] int32
    positions_ptr,  # [max_num_tokens] int64
    is_padding_ptr,  # [max_num_tokens] bool
    pos_in_block_ptr,  # [max_num_tokens] int32
    slot_mappings_ptr,  # [num_groups, max_num_tokens] int64
    slot_stride,
    num_groups: tl.constexpr,
    q_pad,
    PAD_SLOT_ID: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Cap ``query_start_loc`` at ``Q_tot`` and pad ``[Q_tot, q_pad)``."""
    if tl.program_id(0) != 0:
        return
    live = tl.load(live_ptr)
    q_tot = tl.load(q_cu_ptr + live)
    for i in range(0, R + 1, BLOCK):
        block = i + tl.arange(0, BLOCK)
        mask = (block >= live) & (block <= R)
        tl.store(qsl_ptr + block, q_tot, mask=mask)
        seq_mask = (block >= live) & (block < R)
        tl.store(seq_ptr + block, 0, mask=seq_mask)
    for i in range(0, q_pad, BLOCK):
        block = i + tl.arange(0, BLOCK)
        mask = (block >= q_tot) & (block < q_pad)
        tl.store(input_ids_ptr + block, 0, mask=mask)
        tl.store(positions_ptr + block, 0, mask=mask)
        tl.store(is_padding_ptr + block, True, mask=mask)
        tl.store(pos_in_block_ptr + block, 0, mask=mask)
        for g in tl.static_range(0, num_groups):
            tl.store(
                slot_mappings_ptr + g * slot_stride + block, PAD_SLOT_ID, mask=mask
            )


@triton.jit(do_not_specialize=["R"])
def _gpu_observe_kernel(
    slots_ptr,  # [R] int64
    num_sampled_ptr,  # [R] int32
    num_rejected_ptr,  # [R] int32
    conf_ptr,  # [MAX_R, CONF_STRIDE] fp32/bf16
    conf_stride: tl.constexpr,
    gpu_tau_ptr,
    gpu_ema_ptr,
    gpu_hit_ptr,
    gpu_q_ptr,
    gpu_steps_ptr,
    gpu_d_ptr,
    R,
    BLOCK: tl.constexpr,
):
    """Fold verified draft stats into on-device trajectory state in one kernel."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < R
    slot = tl.load(slots_ptr + idx, mask=mask, other=0)
    depth = tl.load(gpu_d_ptr + slot, mask=mask, other=0)
    active = mask & (depth > 0)
    ns = tl.load(num_sampled_ptr + idx, mask=mask, other=0)
    nr = tl.load(num_rejected_ptr + idx, mask=mask, other=0)
    tau = tl.maximum(ns - 1, 0).to(tl.float32)
    ema = tl.load(gpu_ema_ptr + slot, mask=mask, other=0.0)
    df = depth.to(tl.float32)
    keep = active & (tau >= df) & (ema > tau)
    ema_new = tl.where(keep, ema, 0.7 * ema + 0.3 * tau)
    tl.store(gpu_ema_ptr + slot, tl.where(active, ema_new, ema), mask=mask)
    tl.store(
        gpu_tau_ptr + slot,
        tl.where(active, tau, tl.load(gpu_tau_ptr + slot, mask=mask, other=0.0)),
        mask=mask,
    )
    hit = active & (tau >= df) & (nr == 0)
    tl.store(
        gpu_hit_ptr + slot,
        tl.where(
            active,
            hit.to(tl.float32),
            tl.load(gpu_hit_ptr + slot, mask=mask, other=0.0),
        ),
        mask=mask,
    )
    qsum = tl.zeros([BLOCK], dtype=tl.float32)
    qcnt = tl.zeros([BLOCK], dtype=tl.int32)
    for k in tl.static_range(15):
        c = tl.load(
            conf_ptr + slot * conf_stride + k, mask=active & (k < depth), other=0.0
        ).to(tl.float32)
        ok = active & (k < depth) & (c == c) & (c <= 1.0) & (c >= 0.0)
        qsum += tl.where(ok, c, 0.0)
        qcnt += tl.where(ok, 1, 0)
    qmean = tl.where(qcnt > 0, qsum / tl.maximum(qcnt, 1).to(tl.float32), 0.5)
    tl.store(
        gpu_q_ptr + slot,
        tl.where(active, qmean, tl.load(gpu_q_ptr + slot, mask=mask, other=0.5)),
        mask=mask,
    )
    steps = tl.load(gpu_steps_ptr + slot, mask=mask, other=0)
    tl.store(gpu_steps_ptr + slot, tl.where(active, steps + 1, steps), mask=mask)


@triton.jit(do_not_specialize=["R"])
def _gather_aux_ends_kernel(
    aux0_ptr,
    aux1_ptr,
    aux2_ptr,
    aux3_ptr,
    aux4_ptr,
    aux_stride,
    ends_ptr,  # [R] int64
    out_ptr,  # [R, 5 * H]
    out_stride,
    R,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather 5 auxiliary hidden layers at per-request end indices in one kernel."""
    req = tl.program_id(0)
    if req >= R:
        return
    end = tl.load(ends_ptr + req)
    row_in = end * aux_stride
    row_out = req * out_stride
    for off in range(0, H, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < H
        tl.store(
            out_ptr + row_out + 0 * H + cols,
            tl.load(aux0_ptr + row_in + cols, mask=m),
            mask=m,
        )
        tl.store(
            out_ptr + row_out + 1 * H + cols,
            tl.load(aux1_ptr + row_in + cols, mask=m),
            mask=m,
        )
        tl.store(
            out_ptr + row_out + 2 * H + cols,
            tl.load(aux2_ptr + row_in + cols, mask=m),
            mask=m,
        )
        tl.store(
            out_ptr + row_out + 3 * H + cols,
            tl.load(aux3_ptr + row_in + cols, mask=m),
            mask=m,
        )
        tl.store(
            out_ptr + row_out + 4 * H + cols,
            tl.load(aux4_ptr + row_in + cols, mask=m),
            mask=m,
        )


@triton.jit(do_not_specialize=["R"])
def _traj_bind_state_kernel(
    slots_ptr,  # [R] int64
    req_in_ptr,  # [R] int64
    gpu_req_ptr,  # [MAX_R] int64
    gpu_tau_ptr,  # [MAX_R] fp32
    gpu_ema_ptr,  # [MAX_R] fp32
    gpu_hit_ptr,  # [MAX_R] fp32
    gpu_q_ptr,  # [MAX_R] fp32
    gpu_steps_ptr,  # [MAX_R] int32
    gpu_d_ptr,  # [MAX_R] int32
    state_out_ptr,  # [R, 5] fp32
    live_ptr,  # [1] int32 (optional)
    R,
    HAS_LIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Bind request IDs to slots, reset changed slots, and emit [R, 5] gate state."""
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = idx < R
    if HAS_LIVE:
        live = tl.load(live_ptr)
        mask = rmask & (idx < live)
    else:
        mask = rmask
    slot = tl.load(slots_ptr + idx, mask=mask, other=0)
    req = tl.load(req_in_ptr + idx, mask=mask, other=-1)
    prev = tl.load(gpu_req_ptr + slot, mask=mask, other=-1)
    changed = prev != req
    tau = tl.where(changed, 0.0, tl.load(gpu_tau_ptr + slot, mask=mask, other=0.0))
    ema = tl.where(changed, 0.0, tl.load(gpu_ema_ptr + slot, mask=mask, other=0.0))
    hit = tl.where(changed, 0.0, tl.load(gpu_hit_ptr + slot, mask=mask, other=0.0))
    q = tl.where(changed, 0.5, tl.load(gpu_q_ptr + slot, mask=mask, other=0.5))
    steps = tl.where(changed, 0, tl.load(gpu_steps_ptr + slot, mask=mask, other=0))
    d = tl.where(changed, 0, tl.load(gpu_d_ptr + slot, mask=mask, other=0))
    tl.store(gpu_req_ptr + slot, req, mask=mask)
    tl.store(gpu_tau_ptr + slot, tau, mask=mask)
    tl.store(gpu_ema_ptr + slot, ema, mask=mask)
    tl.store(gpu_hit_ptr + slot, hit, mask=mask)
    tl.store(gpu_q_ptr + slot, q, mask=mask)
    tl.store(gpu_steps_ptr + slot, steps, mask=mask)
    tl.store(gpu_d_ptr + slot, d, mask=mask)
    tl.store(state_out_ptr + idx * 5 + 0, tl.where(mask, tau / 15.0, 0.0), mask=rmask)
    tl.store(state_out_ptr + idx * 5 + 1, tl.where(mask, ema / 15.0, 0.0), mask=rmask)
    tl.store(state_out_ptr + idx * 5 + 2, tl.where(mask, hit, 0.0), mask=rmask)
    tl.store(state_out_ptr + idx * 5 + 3, tl.where(mask, q, 0.5), mask=rmask)
    tl.store(
        state_out_ptr + idx * 5 + 4, tl.where(mask & (steps == 0), 1.0, 0.0), mask=rmask
    )


@triton.jit(do_not_specialize=["R", "N", "vfloor_f", "vslope_f"])
def _elastic_auction_kernel(
    sv_ptr,  # [N] fp32 sorted delta
    sver_ptr,  # [N] fp32 sorted verified
    sval_ptr,  # [N] bool/uint8 sorted valid
    order_ptr,  # [N] int64 sorted flat index
    rows_ptr,  # [M] fp32 draft rows
    ms_ptr,  # [M] fp32 draft ms
    vcost_ptr,  # [V_LEN] fp32 verify cost (or dummy if HAS_VCOST=0)
    out_g_ptr,  # [R] int32 (must be zero-initialized)
    out_gu_ptr,  # [R] int32 (uncapped, must be zero-initialized if HAS_GU)
    out_lam_ptr,  # [1] fp32
    decode_ptr,  # [R] bool/uint8
    p_next_ptr,  # [1] fp32
    vfloor_f,
    vslope_f,
    R,
    N,
    M: tl.constexpr,
    V_LEN: tl.constexpr,
    HAS_VCOST: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_Q_ROWS: tl.constexpr = 999999,
    HAS_GU: tl.constexpr = 0,
):
    if tl.program_id(0) != 0:
        return
    idx = tl.arange(0, BLOCK)
    p_next_f = tl.load(p_next_ptr)
    req_mask = idx < R
    r_f = tl.sum(
        tl.where(
            req_mask, tl.load(decode_ptr + idx, mask=req_mask, other=0) != 0, 0
        ).to(tl.float32),
        axis=0,
    )
    mask = idx < N
    sv = tl.load(sv_ptr + idx, mask=mask, other=0.0)
    sver = tl.load(sver_ptr + idx, mask=mask, other=0.0)
    sval = tl.load(sval_ptr + idx, mask=mask, other=0) != 0
    srow = tl.where(sval, 4.0, 0.0)

    c_y = tl.cumsum(sv, axis=0)
    c_ver = tl.cumsum(sver, axis=0)
    c_rows = tl.cumsum(srow, axis=0)

    # Piecewise-linear t_draft(c_rows)
    r0 = tl.load(rows_ptr)
    m0 = tl.load(ms_ptr)
    td = tl.full([BLOCK], m0, dtype=tl.float32)
    td0 = m0
    for j in range(1, M):
        rj = tl.load(rows_ptr + j)
        mj = tl.load(ms_ptr + j)
        in_seg = (c_rows > r0) & (c_rows <= rj)
        span = tl.maximum(rj - r0, 1e-6)
        interp = m0 + (mj - m0) * ((c_rows - r0) / span)
        td = tl.where(in_seg, interp, td)
        if j == M - 1:
            td = tl.where(c_rows > rj, mj, td)
        r0 = rj
        m0 = mj

    if HAS_VCOST:
        n1 = tl.minimum(
            tl.maximum(tl.extra.cuda.libdevice.llrint(r_f + c_ver), 0), V_LEN - 1
        )
        tv1 = tl.load(vcost_ptr + n1)
        n1_0 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.llrint(r_f), 0), V_LEN - 1)
        tv1_0 = tl.load(vcost_ptr + n1_0)
        n2 = tl.minimum(
            tl.maximum(tl.extra.cuda.libdevice.llrint(p_next_f + r_f + c_ver), 0),
            V_LEN - 1,
        )
        tv2 = tl.load(vcost_ptr + n2)
        n2_0 = tl.minimum(
            tl.maximum(tl.extra.cuda.libdevice.llrint(p_next_f + r_f), 0), V_LEN - 1
        )
        tv2_0 = tl.load(vcost_ptr + n2_0)
    else:
        tv1 = vfloor_f + vslope_f * tl.maximum(r_f + c_ver, 0.0)
        tv1_0 = vfloor_f + vslope_f * tl.maximum(r_f, 0.0)
        tv2 = vfloor_f + vslope_f * tl.maximum(p_next_f + r_f + c_ver, 0.0)
        tv2_0 = vfloor_f + vslope_f * tl.maximum(p_next_f + r_f, 0.0)

    feasible_u = mask
    feasible = mask & (c_rows <= MAX_Q_ROWS)
    rate0 = r_f / tl.maximum(td0 + tv1_0, 1e-6)
    rate = tl.where(feasible_u, (r_f + c_y) / tl.maximum(td + tv1, 1e-6), -1e30)
    lam = tl.maximum(rate0, tl.max(rate, axis=0))
    tl.store(out_lam_ptr, lam)

    util0 = -lam * (td0 + tv2_0)
    util_cap = tl.where(feasible, c_y - lam * (td + tv2), -1e30)
    best_u = tl.max(util_cap, axis=0)
    best_idx = tl.argmax(util_cap, axis=0, tie_break_left=True)
    g_tot = tl.where(util0 >= best_u, 0, best_idx + 1)

    take = feasible & sval & (idx < g_tot)
    order = tl.load(order_ptr + idx, mask=mask, other=0)
    req = order // 4
    tl.atomic_add(out_g_ptr + req, 1, mask=take)

    if HAS_GU:
        util_u = tl.where(feasible_u, c_y - lam * (td + tv2), -1e30)
        best_uu = tl.max(util_u, axis=0)
        best_idx_u = tl.argmax(util_u, axis=0, tie_break_left=True)
        g_tot_u = tl.where(util0 >= best_uu, 0, best_idx_u + 1)
        take_u = feasible_u & sval & (idx < g_tot_u)
        tl.atomic_add(out_gu_ptr + req, 1, mask=take_u)


@triton.jit(do_not_specialize=["R"])
def _elastic_cap_publish_kernel(
    g_base_ptr,  # [R] int32 (auction result, already <= MAX_Q_ROWS)
    g_floored_ptr,  # [R] int32 (with hit_cap & cold floors)
    g_uncapped_ptr,  # [R] int32 (uncapped auction + floors)
    slots_ptr,  # [R] int64
    q_out_ptr,  # [R] int32
    d_slot_ptr,  # [MAX_R] int32
    uq_slot_ptr,  # [MAX_R] int32
    live_ptr,  # [1] int32
    R,
    MAX_Q_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Apply hit_cap/cold floors up to MAX_Q_ROWS and publish per-slot d and uncapped q."""
    if tl.program_id(0) != 0:
        return
    live = tl.load(live_ptr)
    idx = tl.arange(0, BLOCK)
    mask = idx < live
    rmask = idx < R
    gb = tl.load(g_base_ptr + idx, mask=mask, other=0)
    gf = tl.load(g_floored_ptr + idx, mask=mask, other=0)
    gu = tl.load(g_uncapped_ptr + idx, mask=mask, other=0)
    extra = tl.maximum(gf - gb, 0) * 4
    used = tl.sum(gb * 4, axis=0)
    room = tl.maximum(MAX_Q_ROWS - used, 0)
    cum_extra = tl.cumsum(extra, axis=0)
    keep_extra = tl.where(cum_extra <= room, extra, 0)
    q_final = tl.where(mask, gb * 4 + keep_extra, 0)
    q_uncapped = gu * 4
    d_final = tl.where(q_final > 0, q_final - 1, 0)
    tl.store(q_out_ptr + idx, q_final, mask=rmask)
    slot = tl.load(slots_ptr + idx, mask=mask, other=0)
    tl.store(d_slot_ptr + slot, d_final, mask=mask)
    tl.store(uq_slot_ptr + slot, q_uncapped, mask=mask)


@triton.jit(do_not_specialize=["num_reqs", "R"])
def _stage_elastic_reqs_kernel(
    src_qsl_ptr,
    dst_qsl_ptr,
    src_idx_ptr,
    dst_idx_ptr,
    src_ns_ptr,
    dst_ns_ptr,
    src_nr_ptr,
    dst_nr_ptr,
    src_ls_ptr,
    dst_ls_ptr,
    src_np_ptr,
    dst_np_ptr,
    src_temp_ptr,
    dst_temp_ptr,
    src_seeds_ptr,
    dst_seeds_ptr,
    num_reqs,
    R,
    BLOCK: tl.constexpr,
):
    """Fuse all 8 per-request elastic stage copies into one kernel launch."""
    idx = tl.arange(0, BLOCK)
    m_qsl = idx < (num_reqs + 1)
    m_req = idx < num_reqs
    m_all = idx < R
    tl.store(
        dst_qsl_ptr + idx,
        tl.load(src_qsl_ptr + idx, mask=m_qsl, other=0).to(tl.int32),
        mask=m_qsl,
    )
    tl.store(
        dst_idx_ptr + idx,
        tl.load(src_idx_ptr + idx, mask=m_req, other=0).to(tl.int64),
        mask=m_req,
    )
    tl.store(
        dst_ns_ptr + idx,
        tl.load(src_ns_ptr + idx, mask=m_req, other=0).to(tl.int32),
        mask=m_req,
    )
    tl.store(
        dst_nr_ptr + idx,
        tl.load(src_nr_ptr + idx, mask=m_req, other=0).to(tl.int32),
        mask=m_req,
    )
    tl.store(
        dst_ls_ptr + idx,
        tl.load(src_ls_ptr + idx, mask=m_all, other=0).to(tl.int64),
        mask=m_all,
    )
    tl.store(
        dst_np_ptr + idx,
        tl.load(src_np_ptr + idx, mask=m_all, other=0).to(tl.int64),
        mask=m_all,
    )
    tl.store(
        dst_temp_ptr + idx,
        tl.load(src_temp_ptr + idx, mask=m_all, other=0.0).to(tl.float32),
        mask=m_all,
    )
    tl.store(
        dst_seeds_ptr + idx,
        tl.load(src_seeds_ptr + idx, mask=m_all, other=0).to(tl.int64),
        mask=m_all,
    )
