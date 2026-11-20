# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-Cost Conditional Tree Verification for CRL (Part 1 of Unified Tree-to-Chain).

Expands CRL's already-computed ``[R, N, K_a, K]`` conditional score table into a
Top-B connected subtree with zero extra drafter forward passes, verifies it with
Split-Tree FA3 + fused Triton intra-tree attention & LSE merge, runs exact lossless
tree verification (identical for T=0 and T>0 on deterministic trees), and compacts
the accepted path's KV cache and hidden states back into contiguous layout.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton


def get_crl_tree_budget() -> int:
    """Return configured CRL tree budget B (0 = disabled)."""
    return int(os.environ.get("VLLM_CRL_TREE_BUDGET", "0"))


def get_crl_tree_temp() -> float:
    """Temperature applied to CRL logits when computing node survival weights."""
    return float(os.environ.get("VLLM_CRL_TREE_TEMP", "1.09"))


# ---------------------------------------------------------------------------
# 1. GPU Top-B Conditional Tree Builder (Triton, 1 program per request)
# ---------------------------------------------------------------------------


@triton.jit
def _build_crl_tree_kernel(
    scores_ptr,  # [num_reqs, N, K, K] float32 (from CRL _score_candidates)
    cand_ids_ptr,  # [num_reqs, N, K] int64
    idx_mapping_ptr,  # [num_reqs * N] int64 (req_state_idx repeated N times)
    # Scratch buffers per req: [max_num_reqs, MAX_FRONTIER]
    f_w_ptr,  # float32
    f_parent_ptr,  # int32
    f_row_ptr,  # int32
    f_rank_ptr,  # int32
    # Outputs indexed by req_state_idx (persistent) AND dense draft_tokens[req, B]
    out_draft_tokens_ptr,  # [max_num_reqs, B] int64 (indexed by req = 0..num_reqs-1)
    out_parents_ptr,  # [max_num_reqs, B+1] int32 (indexed by req_state)
    out_depths_ptr,  # [max_num_reqs, B+1] int32 (indexed by req_state)
    out_mask_ptr,  # [max_num_reqs, B+1, B+1] uint8 (indexed by req_state)
    out_tree_toks_ptr,  # [max_num_reqs, B] int64 (indexed by req_state)
    inv_temp,
    pad_token,
    N_ROWS: tl.constexpr,
    TOP_K: tl.constexpr,
    BUDGET: tl.constexpr,
    MAX_BR: tl.constexpr,
    MAX_FRONTIER: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B1: tl.constexpr,
):
    req = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + req * N_ROWS)
    if req_state < 0:
        return

    B1: tl.constexpr = BUDGET + 1
    offs_f = tl.arange(0, MAX_FRONTIER)
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < TOP_K
    offs_b1 = tl.arange(0, BLOCK_B1)
    b1_mask = offs_b1 < B1

    f_base = req * MAX_FRONTIER
    # Clear frontier weights to -1.0 (inactive)
    tl.store(f_w_ptr + f_base + offs_f, -1.0)

    # Root node 0 (anchor): parent=-1, depth=0, mask[0, 0]=1
    p_base = req_state * B1
    m_base = req_state * B1 * B1
    tl.store(out_parents_ptr + p_base + 0, -1)
    tl.store(out_depths_ptr + p_base + 0, 0)
    root_mask = (offs_b1 == 0).to(tl.uint8)
    tl.store(out_mask_ptr + m_base + offs_b1, root_mask, mask=b1_mask)

    # Compute row 0 softmax probabilities: scores[req, 0, 0, 0..K-1]
    s0_base = (req * N_ROWS * TOP_K + 0) * TOP_K
    s0 = (
        tl.load(scores_ptr + s0_base + offs_k, mask=k_mask, other=float("-inf")).to(
            tl.float32
        )
        * inv_temp
    )
    s0_max = tl.max(s0, axis=0)
    e0 = tl.exp(s0 - s0_max)
    e0 = tl.where(k_mask, e0, 0.0)
    p0 = e0 / tl.sum(e0, axis=0)

    # Push top MAX_BR children of root (row 0) into frontier slots 0..MAX_BR-1
    f_size = 0
    rem_p0 = p0
    for _ in range(MAX_BR):
        best_r = tl.argmax(rem_p0, axis=0)
        best_w = tl.max(rem_p0, axis=0)
        if best_w > 0.0:
            tl.store(f_w_ptr + f_base + f_size, best_w)
            tl.store(f_parent_ptr + f_base + f_size, 0)
            tl.store(f_row_ptr + f_base + f_size, 0)
            tl.store(f_rank_ptr + f_base + f_size, best_r)
            f_size += 1
            rem_p0 = tl.where(offs_k == best_r, -1.0, rem_p0)

    # Iteratively pop highest-weight frontier node for node_id = 1..BUDGET
    for node_id in range(1, B1):
        fw = tl.load(f_w_ptr + f_base + offs_f)
        best_f = tl.argmax(fw, axis=0)
        best_w = tl.max(fw, axis=0)

        if best_w <= 0.0:
            # Degenerate fallback: attach pad_token to root
            tl.store(out_draft_tokens_ptr + req * BUDGET + (node_id - 1), pad_token)
            tl.store(out_tree_toks_ptr + req_state * BUDGET + (node_id - 1), pad_token)
            tl.store(out_parents_ptr + p_base + node_id, 0)
            tl.store(out_depths_ptr + p_base + node_id, 1)
            row_m = ((offs_b1 == 0) | (offs_b1 == node_id)).to(tl.uint8)
            tl.store(
                out_mask_ptr + m_base + node_id * B1 + offs_b1, row_m, mask=b1_mask
            )
        else:
            # Deactivate popped frontier slot
            tl.store(f_w_ptr + f_base + best_f, -1.0)
            par_id = tl.load(f_parent_ptr + f_base + best_f)
            row = tl.load(f_row_ptr + f_base + best_f)
            rank = tl.load(f_rank_ptr + f_base + best_f)

            tok = tl.load(cand_ids_ptr + (req * N_ROWS + row) * TOP_K + rank)
            tl.store(out_draft_tokens_ptr + req * BUDGET + (node_id - 1), tok)
            tl.store(out_tree_toks_ptr + req_state * BUDGET + (node_id - 1), tok)
            tl.store(out_parents_ptr + p_base + node_id, par_id)
            tl.store(out_depths_ptr + p_base + node_id, row + 1)

            # Tree mask row: copy parent's mask row and set self bit
            par_m = tl.load(
                out_mask_ptr + m_base + par_id * B1 + offs_b1, mask=b1_mask, other=0
            )
            cur_m = tl.where(offs_b1 == node_id, 1, par_m).to(tl.uint8)
            tl.store(
                out_mask_ptr + m_base + node_id * B1 + offs_b1, cur_m, mask=b1_mask
            )

            # Expand children into next row if row + 1 < N_ROWS and frontier has room
            next_row = row + 1
            if (next_row < N_ROWS) & (f_size + MAX_BR <= MAX_FRONTIER):
                sn_base = ((req * N_ROWS + next_row) * TOP_K + rank) * TOP_K
                sn = (
                    tl.load(
                        scores_ptr + sn_base + offs_k, mask=k_mask, other=float("-inf")
                    ).to(tl.float32)
                    * inv_temp
                )
                sn_max = tl.max(sn, axis=0)
                en = tl.exp(sn - sn_max)
                en = tl.where(k_mask, en, 0.0)
                pn = en / tl.sum(en, axis=0)
                rem_pn = pn
                for _ in range(MAX_BR):
                    br = tl.argmax(rem_pn, axis=0)
                    bw = tl.max(rem_pn, axis=0)
                    wn = best_w * bw
                    if wn > 0.0:
                        tl.store(f_w_ptr + f_base + f_size, wn)
                        tl.store(f_parent_ptr + f_base + f_size, node_id)
                        tl.store(f_row_ptr + f_base + f_size, next_row)
                        tl.store(f_rank_ptr + f_base + f_size, br)
                        f_size += 1
                        rem_pn = tl.where(offs_k == br, -1.0, rem_pn)


# ---------------------------------------------------------------------------
# 2. Fused Intra-Tree Attention + LSE Merge (with native FA3 Prefix)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_intra_tree_and_merge_varlen_kernel(
    q_ptr,  # [T, H, D] bf16/fp16 (may be non-contiguous slice of qkv!)
    k_ptr,  # [T, H_kv, D]
    v_ptr,  # [T, H_kv, D]
    out_p_ptr,  # [T, H, D] (prefix FA3 output)
    lse_p_ptr,  # [H, T] fp32 (prefix FA3 LSE)
    mask_ptr,  # [num_reqs, MAX_Q, MAX_Q] uint8
    cu_q_ptr,  # [num_reqs + 1] int32
    prefix_lens_ptr,  # [num_reqs] int32
    out_ptr,  # [T, H, D]
    scale,
    q_s0,
    q_s1,
    k_s0,
    k_s1,
    v_s0,
    v_s1,
    op_s0,
    op_s1,
    lp_s0,
    lp_s1,
    o_s0,
    o_s1,
    MAX_Q: tl.constexpr,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    bh = tl.program_id(0)
    qi = tl.program_id(1)
    b = bh // H
    h = bh % H
    h_kv = h // (H // H_kv)

    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    q_len = q_end - q_start
    if qi >= q_len:
        return

    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, BLOCK_Q)
    k_valid = offs_k < q_len

    tok_q = q_start + qi
    q = tl.load(q_ptr + tok_q * q_s0 + h * q_s1 + offs_d).to(tl.float32)

    m_row = tl.load(mask_ptr + (b * MAX_Q + qi) * MAX_Q + offs_k, mask=k_valid, other=0)
    allowed = k_valid & (m_row != 0)

    tok_k = q_start + offs_k
    k_ptrs = k_ptr + tok_k[:, None] * k_s0 + h_kv * k_s1 + offs_d[None, :]
    k = tl.load(k_ptrs, mask=allowed[:, None], other=0.0).to(tl.float32)

    scores = tl.sum(k * q[None, :], axis=1) * scale
    scores = tl.where(allowed, scores, float("-inf"))

    max_s = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - max_s)
    sum_e = tl.sum(exp_s, axis=0)
    lse_t = max_s + tl.log(sum_e)
    probs = exp_s / sum_e

    v_ptrs = v_ptr + tok_k[:, None] * v_s0 + h_kv * v_s1 + offs_d[None, :]
    v = tl.load(v_ptrs, mask=allowed[:, None], other=0.0).to(tl.float32)
    out_t = tl.sum(v * probs[:, None], axis=0)

    pref_len = tl.load(prefix_lens_ptr + b)
    if pref_len > 0:
        out_p = tl.load(out_p_ptr + tok_q * op_s0 + h * op_s1 + offs_d).to(tl.float32)
        lse_p = tl.load(lse_p_ptr + h * lp_s0 + tok_q * lp_s1).to(tl.float32)
        m = tl.maximum(lse_p, lse_t)
        ep = tl.exp(lse_p - m)
        et = tl.exp(lse_t - m)
        out_merged = (out_p * ep + out_t * et) / (ep + et)
    else:
        out_merged = out_t

    tl.store(
        out_ptr + tok_q * o_s0 + h * o_s1 + offs_d,
        out_merged.to(q_ptr.dtype.element_ty),
    )


def fused_intra_tree_and_merge_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_p: torch.Tensor,
    lse_p: torch.Tensor,
    tree_mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    num_reqs: int,
    max_q: int,
) -> None:
    _, H, D = q.shape
    H_kv = k.shape[1]
    BLOCK_Q = triton.next_power_of_2(max_q)
    grid = (num_reqs * H, max_q)
    _fused_intra_tree_and_merge_varlen_kernel[grid](
        q,
        k,
        v,
        out_p,
        lse_p,
        tree_mask,
        cu_seqlens_q,
        prefix_lens,
        out,
        scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out_p.stride(0),
        out_p.stride(1),
        lse_p.stride(0),
        lse_p.stride(1),
        out.stride(0),
        out.stride(1),
        MAX_Q=max_q,
        H=H,
        H_kv=H_kv,
        D=D,
        BLOCK_Q=BLOCK_Q,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# 3. Tree Position Assignment + Step Mask Gather (CUDA-Graph safe)
# ---------------------------------------------------------------------------


@triton.jit
def _prepare_tree_step_kernel(
    idx_mapping_ptr,  # [num_reqs] int64
    query_start_loc_ptr,  # [num_reqs + 1] int32
    num_computed_ptr,  # [max_num_reqs] int32 (prefix_len BEFORE this step)
    is_tree_req_ptr,  # [num_reqs] uint8 (1 if req has tree drafts, 0 if prefill)
    tree_depths_ptr,  # [max_num_reqs, B1] int32
    tree_mask_ptr,  # [max_num_reqs, B1, B1] uint8
    positions_ptr,  # [num_tokens] int64 (in/out)
    step_prefix_lens_ptr,  # [max_num_reqs] int32 (out)
    step_tree_mask_ptr,  # [max_num_reqs, B1, B1] uint8 (out)
    B1: tl.constexpr,
    BLOCK_B1: tl.constexpr,
):
    r = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + r)
    pref_len = tl.load(num_computed_ptr + req_state)
    tl.store(step_prefix_lens_ptr + r, pref_len)

    if tl.load(is_tree_req_ptr + r) == 0:
        return

    q_start = tl.load(query_start_loc_ptr + r)
    q_end = tl.load(query_start_loc_ptr + r + 1)
    q_len = q_end - q_start

    offs = tl.arange(0, BLOCK_B1)
    valid = offs < q_len

    # Overwrite RoPE positions for tree nodes: prefix_len + tree_depth[j]
    depths = tl.load(tree_depths_ptr + req_state * B1 + offs, mask=valid, other=0)
    tl.store(
        positions_ptr + q_start + offs, (pref_len + depths).to(tl.int64), mask=valid
    )

    # Copy tree_mask[req_state] -> step_tree_mask[r]
    src_m = tree_mask_ptr + req_state * B1 * B1
    dst_m = step_tree_mask_ptr + r * B1 * B1
    b1_mask = offs < B1
    for row in range(B1):
        vals = tl.load(src_m + row * B1 + offs, mask=b1_mask, other=0)
        tl.store(dst_m + row * B1 + offs, vals, mask=b1_mask)


# ---------------------------------------------------------------------------
# 4. Exact Lossless Tree Verification Kernel (identical for T=0 and T>0!)
# ---------------------------------------------------------------------------


@triton.jit
def _tree_verify_kernel(
    target_tokens_ptr,  # [total_logits] int64 (sampled from target logits at each tree node)
    idx_mapping_ptr,  # [num_reqs] int64
    cu_num_logits_ptr,  # [num_reqs + 1] int32
    tree_tokens_ptr,  # [max_num_reqs, B] int64
    tree_parents_ptr,  # [max_num_reqs, B1] int32
    out_sampled_ptr,  # [num_reqs, B1] int64 (-1 padded)
    out_acc_nodes_ptr,  # [num_reqs, B1] int32 (accepted node indices: [0, n_1, ..., n_k])
    out_num_sampled_ptr,  # [num_reqs] int32
    out_num_rejected_ptr,  # [num_reqs] int32
    BUDGET: tl.constexpr,
    B1: tl.constexpr,
):
    r = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + r)
    l_start = tl.load(cu_num_logits_ptr + r)
    l_end = tl.load(cu_num_logits_ptr + r + 1)
    n_logits = l_end - l_start

    # Initialize output row to -1
    for j in range(B1):
        tl.store(out_sampled_ptr + r * B1 + j, -1)
        tl.store(out_acc_nodes_ptr + r * B1 + j, 0)

    if n_logits <= 0:
        tl.store(out_num_sampled_ptr + r, 0)
        tl.store(out_num_rejected_ptr + r, 0)
        return

    if n_logits == 1:
        # Non-speculative request (e.g., prefill-only): accept single sampled token
        tok0 = tl.load(target_tokens_ptr + l_start)
        tl.store(out_sampled_ptr + r * B1 + 0, tok0)
        tl.store(out_acc_nodes_ptr + r * B1 + 0, 0)
        tl.store(out_num_sampled_ptr + r, 1)
        tl.store(out_num_rejected_ptr + r, 0)
        return

    # Walk deterministic tree starting at root u = 0
    u = 0
    step = 0
    p_base = req_state * B1
    t_base = req_state * BUDGET
    keep_going = True

    for _ in range(B1):
        if keep_going:
            t_star = tl.load(target_tokens_ptr + l_start + u)
            tl.store(out_sampled_ptr + r * B1 + step, t_star)
            tl.store(out_acc_nodes_ptr + r * B1 + step, u)
            step += 1

            # Search children v in (u+1 .. n_logits-1) with parent[v] == u and tok[v-1] == t_star
            matched_v = -1
            for v in range(1, B1):
                if (v < n_logits) & (matched_v < 0):
                    pv = tl.load(tree_parents_ptr + p_base + v)
                    if pv == u:
                        tv = tl.load(tree_tokens_ptr + t_base + (v - 1))
                        if tv == t_star:
                            matched_v = v
            if matched_v >= 0:
                u = matched_v
            else:
                keep_going = False

    tl.store(out_num_sampled_ptr + r, step)
    tl.store(out_num_rejected_ptr + r, n_logits - step)


# ---------------------------------------------------------------------------
# 5. In-Place Accepted-Path KV Cache & Hidden-State Compaction
# ---------------------------------------------------------------------------


@triton.jit
def _compact_kv_cache_4d_batched_kernel(
    kv_ptrs_ptr,  # [num_layers] int64 device pointers to each layer's kv_cache
    block_table_ptr,  # [num_reqs, max_blocks] int32
    prefix_lens_ptr,  # [num_reqs] int32
    acc_nodes_ptr,  # [num_reqs, B1] int32
    num_sampled_ptr,  # [num_reqs] int32
    block_table_stride: tl.constexpr,
    kv_stride_blk,
    kv_stride_h,
    kv_stride_pos,
    H_KV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TWO_D: tl.constexpr,
    B1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (num_reqs, num_layers * H_KV)
    r = tl.program_id(0)
    lh = tl.program_id(1)
    layer_idx = lh // H_KV
    h = lh % H_KV

    n_samp = tl.load(num_sampled_ptr + r)
    if n_samp <= 1:
        return

    kv_ptr = tl.load(kv_ptrs_ptr + layer_idx).to(tl.pointer_type(tl.bfloat16))
    pref = tl.load(prefix_lens_ptr + r)
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < TWO_D
    base_h = kv_ptr + h * kv_stride_h

    for j in range(1, B1):
        if j < n_samp:
            src_node = tl.load(acc_nodes_ptr + r * B1 + j)
            if src_node != j:
                src_pos = pref + src_node
                dst_pos = pref + j
                src_blk = tl.load(
                    block_table_ptr + r * block_table_stride + src_pos // BLOCK_SIZE
                )
                dst_blk = tl.load(
                    block_table_ptr + r * block_table_stride + dst_pos // BLOCK_SIZE
                )
                src_off = src_pos % BLOCK_SIZE
                dst_off = dst_pos % BLOCK_SIZE
                val = tl.load(
                    base_h + src_blk * kv_stride_blk + src_off * kv_stride_pos + offs,
                    mask=d_mask,
                )
                tl.store(
                    base_h + dst_blk * kv_stride_blk + dst_off * kv_stride_pos + offs,
                    val,
                    mask=d_mask,
                )


@triton.jit
def _compact_hidden_batched_kernel(
    h_ptrs_ptr,  # [num_tensors] int64 device pointers
    query_start_loc_ptr,  # [num_reqs + 1] int32
    acc_nodes_ptr,  # [num_reqs, B1] int32
    num_sampled_ptr,  # [num_reqs] int32
    H_DIM: tl.constexpr,
    B1: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Grid: (num_reqs, num_tensors * cdiv(H_DIM, BLOCK_H))
    r = tl.program_id(0)
    tc = tl.program_id(1)
    n_chunks = tl.cdiv(H_DIM, BLOCK_H)
    t_idx = tc // n_chunks
    chunk = tc % n_chunks

    n_samp = tl.load(num_sampled_ptr + r)
    if n_samp <= 1:
        return
    h_ptr = tl.load(h_ptrs_ptr + t_idx).to(tl.pointer_type(tl.bfloat16))
    q_start = tl.load(query_start_loc_ptr + r)
    offs = chunk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs < H_DIM
    for j in range(1, B1):
        if j < n_samp:
            src_node = tl.load(acc_nodes_ptr + r * B1 + j)
            if src_node != j:
                val = tl.load(h_ptr + (q_start + src_node) * H_DIM + offs, mask=h_mask)
                tl.store(h_ptr + (q_start + j) * H_DIM + offs, val, mask=h_mask)


@triton.jit
def _compact_kv_cache_4d_kernel(
    kv_cache_ptr,  # [num_blocks, H_kv, block_size, 2 * D]
    block_table_ptr,  # [num_reqs, max_blocks] int32
    prefix_lens_ptr,  # [num_reqs] int32
    acc_nodes_ptr,  # [num_reqs, B1] int32
    num_sampled_ptr,  # [num_reqs] int32
    block_table_stride: tl.constexpr,
    kv_stride_blk,
    kv_stride_h,
    kv_stride_pos,
    BLOCK_SIZE: tl.constexpr,
    TWO_D: tl.constexpr,
    B1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (num_reqs, H_kv)
    r = tl.program_id(0)
    h = tl.program_id(1)

    n_samp = tl.load(num_sampled_ptr + r)
    if n_samp <= 1:
        return

    pref = tl.load(prefix_lens_ptr + r)
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < TWO_D
    base_h = kv_cache_ptr + h * kv_stride_h

    # Topological order guarantees acc_nodes[r, j] >= j, so ascending j = 1..n_samp-1 is race-free!
    for j in range(1, B1):
        if j < n_samp:
            src_node = tl.load(acc_nodes_ptr + r * B1 + j)
            if src_node != j:
                src_pos = pref + src_node
                dst_pos = pref + j
                src_blk = tl.load(
                    block_table_ptr + r * block_table_stride + src_pos // BLOCK_SIZE
                )
                dst_blk = tl.load(
                    block_table_ptr + r * block_table_stride + dst_pos // BLOCK_SIZE
                )
                src_off = src_pos % BLOCK_SIZE
                dst_off = dst_pos % BLOCK_SIZE
                val = tl.load(
                    base_h + src_blk * kv_stride_blk + src_off * kv_stride_pos + offs,
                    mask=d_mask,
                )
                tl.store(
                    base_h + dst_blk * kv_stride_blk + dst_off * kv_stride_pos + offs,
                    val,
                    mask=d_mask,
                )


@triton.jit
def _compact_kv_cache_kernel(
    kv_cache_ptr,  # [2, num_blocks, block_size, H_kv, D] (or [num_blocks, 2, ...])
    block_table_ptr,  # [num_reqs, max_blocks] int32
    prefix_lens_ptr,  # [num_reqs] int32
    acc_nodes_ptr,  # [num_reqs, B1] int32
    num_sampled_ptr,  # [num_reqs] int32
    block_table_stride: tl.constexpr,
    kv_stride_2,  # stride between K and V
    kv_stride_blk,  # stride between blocks
    kv_stride_pos,  # stride between positions inside a block
    BLOCK_SIZE: tl.constexpr,
    KV_ELEMS: tl.constexpr,  # H_kv * D
    B1: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    # Grid: (num_reqs, 2 * cdiv(KV_ELEMS, BLOCK_E))
    r = tl.program_id(0)
    pid1 = tl.program_id(1)
    n_chunks = tl.cdiv(KV_ELEMS, BLOCK_E)
    kv_idx = pid1 // n_chunks  # 0 = K, 1 = V
    chunk = pid1 % n_chunks

    n_samp = tl.load(num_sampled_ptr + r)
    if n_samp <= 1:
        return

    pref = tl.load(prefix_lens_ptr + r)
    offs = chunk * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = offs < KV_ELEMS
    base_kv = kv_cache_ptr + kv_idx * kv_stride_2

    # Topological order guarantees acc_nodes[r, j] >= j, so ascending j = 1..n_samp-1 is race-free!
    for j in range(1, B1):
        if j < n_samp:
            src_node = tl.load(acc_nodes_ptr + r * B1 + j)
            if src_node != j:
                src_pos = pref + src_node
                dst_pos = pref + j
                src_blk = tl.load(
                    block_table_ptr + r * block_table_stride + src_pos // BLOCK_SIZE
                )
                dst_blk = tl.load(
                    block_table_ptr + r * block_table_stride + dst_pos // BLOCK_SIZE
                )
                src_off = src_pos % BLOCK_SIZE
                dst_off = dst_pos % BLOCK_SIZE
                val = tl.load(
                    base_kv + src_blk * kv_stride_blk + src_off * kv_stride_pos + offs,
                    mask=e_mask,
                )
                tl.store(
                    base_kv + dst_blk * kv_stride_blk + dst_off * kv_stride_pos + offs,
                    val,
                    mask=e_mask,
                )


@triton.jit
def _compact_hidden_kernel(
    hidden_ptr,  # [num_tokens, H_dim]
    query_start_loc_ptr,  # [num_reqs + 1] int32
    acc_nodes_ptr,  # [num_reqs, B1] int32
    num_sampled_ptr,  # [num_reqs] int32
    H_DIM: tl.constexpr,
    B1: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    r = tl.program_id(0)
    chunk = tl.program_id(1)
    n_samp = tl.load(num_sampled_ptr + r)
    if n_samp <= 1:
        return
    q_start = tl.load(query_start_loc_ptr + r)
    offs = chunk * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs < H_DIM
    for j in range(1, B1):
        if j < n_samp:
            src_node = tl.load(acc_nodes_ptr + r * B1 + j)
            if src_node != j:
                val = tl.load(
                    hidden_ptr + (q_start + src_node) * H_DIM + offs, mask=h_mask
                )
                tl.store(hidden_ptr + (q_start + j) * H_DIM + offs, val, mask=h_mask)


# ---------------------------------------------------------------------------
# 6. Persistent State Manager
# ---------------------------------------------------------------------------


class CRLTreeState:
    """Persistent GPU buffers and helpers for CRL Zero-Cost Conditional Tree."""

    def __init__(
        self,
        max_num_reqs: int,
        budget: int,
        draft_rows: int,
        top_k: int,
        device: torch.device,
        max_branch: int = 4,
        temp: float = 1.09,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.budget = budget
        self.b1 = budget + 1
        self.draft_rows = draft_rows
        self.top_k = top_k
        self.max_branch = max_branch
        self.temp = temp
        self.device = device
        # Max frontier size: at most (budget + 1) * max_branch, rounded to power of 2
        self.max_frontier = triton.next_power_of_2((budget + 2) * max_branch)

        # Persistent per-req_state tree topology
        self.tree_tokens = torch.zeros(
            max_num_reqs, budget, dtype=torch.int64, device=device
        )
        self.tree_parents = torch.full(
            (max_num_reqs, self.b1), -1, dtype=torch.int32, device=device
        )
        self.tree_depths = torch.zeros(
            max_num_reqs, self.b1, dtype=torch.int32, device=device
        )
        self.tree_mask = torch.zeros(
            max_num_reqs, self.b1, self.b1, dtype=torch.uint8, device=device
        )
        # Initialize default chain topology so dummy / uninitialized slots are valid lower-triangular
        for i in range(self.b1):
            self.tree_parents[:, i] = i - 1
            self.tree_depths[:, i] = i
            self.tree_mask[:, i, : i + 1] = 1

        # Scratch buffers for _build_crl_tree_kernel
        self._f_w = torch.empty(
            max_num_reqs, self.max_frontier, dtype=torch.float32, device=device
        )
        self._f_parent = torch.empty(
            max_num_reqs, self.max_frontier, dtype=torch.int32, device=device
        )
        self._f_row = torch.empty(
            max_num_reqs, self.max_frontier, dtype=torch.int32, device=device
        )
        self._f_rank = torch.empty(
            max_num_reqs, self.max_frontier, dtype=torch.int32, device=device
        )

        # Fixed-address per-step buffers (for CUDA Graph capture & replay)
        self.step_prefix_lens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.step_is_tree_req = torch.ones(
            max_num_reqs, dtype=torch.uint8, device=device
        )
        self.step_tree_mask = torch.zeros(
            max_num_reqs, self.b1, self.b1, dtype=torch.uint8, device=device
        )
        for i in range(self.b1):
            self.step_tree_mask[:, i, : i + 1] = 1

        # Verification output buffers
        self.step_acc_nodes = torch.zeros(
            max_num_reqs, self.b1, dtype=torch.int32, device=device
        )
        # Pre-allocated prefix attention output buffer (lazy-sized on first use)
        self._prefix_out_buf: torch.Tensor | None = None
        # Cached device pointer table for single-launch batched 36-layer KV compaction
        self._kv_ptrs_cache: tuple[int, torch.Tensor] | None = None

    def get_prefix_out_buf(self, like: torch.Tensor) -> torch.Tensor:
        if (
            self._prefix_out_buf is None
            or self._prefix_out_buf.shape[0] < like.shape[0]
            or self._prefix_out_buf.shape[1:] != like.shape[1:]
            or self._prefix_out_buf.dtype != like.dtype
        ):
            self._prefix_out_buf = torch.empty_like(like)
        return self._prefix_out_buf[: like.shape[0]]

    def build_trees(
        self,
        candidate_ids: torch.Tensor,  # [num_reqs, N, K]
        scores: torch.Tensor,  # [num_reqs, N, K, K]
        num_reqs: int,
        sample_idx_mapping: torch.Tensor,  # [num_reqs * N]
        out_draft_tokens: torch.Tensor,  # [max_num_reqs, B]
        pad_token: int = 0,
    ) -> None:
        BLOCK_K = triton.next_power_of_2(self.top_k)
        BLOCK_B1 = triton.next_power_of_2(self.b1)
        _build_crl_tree_kernel[(num_reqs,)](
            scores,
            candidate_ids,
            sample_idx_mapping,
            self._f_w,
            self._f_parent,
            self._f_row,
            self._f_rank,
            out_draft_tokens,
            self.tree_parents,
            self.tree_depths,
            self.tree_mask,
            self.tree_tokens,
            1.0 / self.temp,
            pad_token,
            N_ROWS=self.draft_rows,
            TOP_K=self.top_k,
            BUDGET=self.budget,
            MAX_BR=self.max_branch,
            MAX_FRONTIER=self.max_frontier,
            BLOCK_K=BLOCK_K,
            BLOCK_B1=BLOCK_B1,
            num_warps=4,
        )

    def prepare_step(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
        is_tree_req_np=None,
    ) -> None:
        if is_tree_req_np is not None:
            self.step_is_tree_req[:num_reqs].copy_(
                torch.from_numpy(is_tree_req_np), non_blocking=True
            )
        else:
            self.step_is_tree_req[:num_reqs].fill_(1)
        BLOCK_B1 = triton.next_power_of_2(self.b1)
        _prepare_tree_step_kernel[(num_reqs,)](
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            self.step_is_tree_req,
            self.tree_depths,
            self.tree_mask,
            positions,
            self.step_prefix_lens,
            self.step_tree_mask,
            B1=self.b1,
            BLOCK_B1=BLOCK_B1,
            num_warps=2,
        )

    def verify_trees(
        self,
        target_tokens: torch.Tensor,  # [total_logits] int64
        idx_mapping: torch.Tensor,  # [num_reqs] int64
        cu_num_logits: torch.Tensor,  # [num_reqs + 1] int32
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_sampled = torch.full(
            (num_reqs, self.b1), -1, dtype=torch.int64, device=self.device
        )
        out_num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        out_num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        acc_nodes = self.step_acc_nodes[:num_reqs]
        _tree_verify_kernel[(num_reqs,)](
            target_tokens,
            idx_mapping,
            cu_num_logits,
            self.tree_tokens,
            self.tree_parents,
            out_sampled,
            acc_nodes,
            out_num_sampled,
            out_num_rejected,
            BUDGET=self.budget,
            B1=self.b1,
            num_warps=1,
        )
        return out_sampled, acc_nodes, out_num_sampled, out_num_rejected

    def compact_kv_and_hidden(
        self,
        kv_caches: list[torch.Tensor],
        block_table: torch.Tensor,  # [num_reqs, max_blocks] int32
        num_reqs: int,
        acc_nodes: torch.Tensor,  # [num_reqs, B1] int32
        num_sampled: torch.Tensor,  # [num_reqs] int32
        query_start_loc: torch.Tensor,  # [num_reqs + 1] int32
        draft_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ) -> None:
        BLOCK_E = 256
        if (
            kv_caches
            and kv_caches[0].ndim == 4
            and kv_caches[0].dtype == torch.bfloat16
        ):
            # Single-launch batched compaction across all target layers (zero per-step H2D sync).
            kv0 = kv_caches[0]
            if self._kv_ptrs_cache is None or self._kv_ptrs_cache[0] != id(kv0):
                ptrs = torch.tensor(
                    [c.data_ptr() for c in kv_caches],
                    dtype=torch.int64,
                    device=self.device,
                )
                self._kv_ptrs_cache = (id(kv0), ptrs)
            h_kv = kv0.shape[1]
            block_size = kv0.shape[2]
            two_d = kv0.shape[3]
            BLOCK_D = triton.next_power_of_2(two_d)
            _compact_kv_cache_4d_batched_kernel[(num_reqs, len(kv_caches) * h_kv)](
                self._kv_ptrs_cache[1],
                block_table,
                self.step_prefix_lens,
                acc_nodes,
                num_sampled,
                block_table_stride=block_table.stride(0),
                kv_stride_blk=kv0.stride(0),
                kv_stride_h=kv0.stride(1),
                kv_stride_pos=kv0.stride(2),
                H_KV=h_kv,
                BLOCK_SIZE=block_size,
                TWO_D=two_d,
                B1=self.b1,
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
        else:
            for kv_cache in kv_caches:
                if kv_cache.ndim == 4:
                    h_kv = kv_cache.shape[1]
                    block_size = kv_cache.shape[2]
                    two_d = kv_cache.shape[3]
                    BLOCK_D = triton.next_power_of_2(two_d)
                    _compact_kv_cache_4d_kernel[(num_reqs, h_kv)](
                        kv_cache,
                        block_table,
                        self.step_prefix_lens,
                        acc_nodes,
                        num_sampled,
                        block_table_stride=block_table.stride(0),
                        kv_stride_blk=kv_cache.stride(0),
                        kv_stride_h=kv_cache.stride(1),
                        kv_stride_pos=kv_cache.stride(2),
                        BLOCK_SIZE=block_size,
                        TWO_D=two_d,
                        B1=self.b1,
                        BLOCK_D=BLOCK_D,
                        num_warps=4,
                    )
                elif kv_cache.ndim == 5 and kv_cache.shape[0] == 2:
                    block_size = kv_cache.shape[2]
                    kv_elems = kv_cache.shape[3] * kv_cache.shape[4]
                    n_chunks = triton.cdiv(kv_elems, BLOCK_E)
                    _compact_kv_cache_kernel[(num_reqs, 2 * n_chunks)](
                        kv_cache,
                        block_table,
                        self.step_prefix_lens,
                        acc_nodes,
                        num_sampled,
                        block_table_stride=block_table.stride(0),
                        kv_stride_2=kv_cache.stride(0),
                        kv_stride_blk=kv_cache.stride(1),
                        kv_stride_pos=kv_cache.stride(2),
                        BLOCK_SIZE=block_size,
                        KV_ELEMS=kv_elems,
                        B1=self.b1,
                        BLOCK_E=BLOCK_E,
                        num_warps=4,
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported KV cache shape for tree compaction: {tuple(kv_cache.shape)}"
                    )

        BLOCK_H = 512
        if draft_hidden_states is not None:
            h_dim = draft_hidden_states.shape[-1]
            _compact_hidden_kernel[(num_reqs, triton.cdiv(h_dim, BLOCK_H))](
                draft_hidden_states,
                query_start_loc,
                acc_nodes,
                num_sampled,
                H_DIM=h_dim,
                B1=self.b1,
                BLOCK_H=BLOCK_H,
                num_warps=4,
            )
        if aux_hidden_states is not None:
            for aux in aux_hidden_states:
                h_dim = aux.shape[-1]
                _compact_hidden_kernel[(num_reqs, triton.cdiv(h_dim, BLOCK_H))](
                    aux,
                    query_start_loc,
                    acc_nodes,
                    num_sampled,
                    H_DIM=h_dim,
                    B1=self.b1,
                    BLOCK_H=BLOCK_H,
                    num_warps=4,
                )
