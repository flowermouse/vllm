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
from vllm.v1.worker.gpu.sample.gumbel import _DRAFT_NOISE_SALT, murmur3_uniform32


def get_crl_tree_budget() -> int:
    """Return configured CRL tree budget B (0 = disabled)."""
    return int(os.environ.get("VLLM_CRL_TREE_BUDGET", "0"))


def get_crl_tree_temp() -> float:
    """Temperature applied to CRL logits when computing node survival weights."""
    return float(os.environ.get("VLLM_CRL_TREE_TEMP", "1.09"))


def get_crl_tree_stochastic_verify() -> bool:
    """Whether T > 0 tree requests use stochastic without-replacement drafting
    and recursive residual tree rejection sampling (default True; set 0 for
    deterministic strict-match A/B fallback)."""
    return os.environ.get("VLLM_CRL_TREE_STOCHASTIC_VERIFY", "1") == "1"


def get_crl_tree_stoch_alpha_base() -> float:
    """Base row-level acceptance estimate for rank-0 stochastic tree child
    (alpha_1 = (base + (1 - base) * q_top) * recall_u). Default 0.50 gives the
    exact midpoint (1 + q_top) / 2; set 0.0 for raw q_top weights."""
    return float(os.environ.get("VLLM_CRL_TREE_STOCH_ALPHA_BASE", "0.50"))


def get_crl_tree_single_pass() -> bool:
    """Whether to use the unified single-pass Split-KV Triton tree attention kernel
    (default True; set 0 to force 2-pass Split-Tree FA3 + Triton merge)."""
    return os.environ.get("VLLM_CRL_TREE_SINGLE_PASS", "1") == "1"


def get_crl_tree_single_pass_max_ctx() -> int:
    """Max prefix context length for single-pass Triton tree attention at high
    concurrency (num_reqs >= 32). Beyond this threshold, falls back to 2-pass
    FA3 prefix + Triton intra-tree merge to preserve FA3 TMA long-KV bandwidth."""
    return int(os.environ.get("VLLM_CRL_TREE_SINGLE_PASS_MAX_CTX", "2048"))


# ---------------------------------------------------------------------------
# 1. GPU Top-B Conditional Tree Builder (Triton, 1 program per request)
# ---------------------------------------------------------------------------


@triton.jit
def _build_crl_tree_kernel(
    scores_ptr,  # [num_reqs, N, K, K] float32 (from CRL _score_candidates)
    cand_ids_ptr,  # [num_reqs, N, K] int64
    idx_mapping_ptr,  # [num_reqs * N] int64 (req_state_idx repeated N times)
    null_delta_ptr,  # [max_num_reqs, N, K_A] float32 (when HAS_NULL=True)
    null_beta_ptr,  # [N] float32 (when HAS_NULL=True)
    temperature_ptr,  # [max_num_reqs] float32 (when STOCHASTIC_ENABLED=True)
    seeds_ptr,  # [max_num_reqs] int64 (when STOCHASTIC_ENABLED=True)
    sample_pos_ptr,  # [num_reqs * N] int64 (when STOCHASTIC_ENABLED=True)
    # Scratch buffers per req: [max_num_reqs, MAX_FRONTIER]
    f_w_ptr,  # float32
    f_parent_ptr,  # int32
    f_row_ptr,  # int32
    f_rank_ptr,  # int32
    # Outputs indexed by req_state_idx (persistent) AND dense draft_tokens[req, B]
    out_draft_tokens_ptr,  # [max_num_reqs, B] int64 (indexed by req = 0..num_reqs-1)
    out_node_weights_ptr,  # [max_num_reqs, B] float32 (indexed by req = 0..num_reqs-1)
    out_node_weights_stride,
    out_parents_ptr,  # [max_num_reqs, B+1] int32 (indexed by req_state)
    out_depths_ptr,  # [max_num_reqs, B+1] int32 (indexed by req_state)
    out_mask_ptr,  # [max_num_reqs, B+1, B+1] uint8 (indexed by req_state)
    out_tree_toks_ptr,  # [max_num_reqs, B] int64 (indexed by req_state)
    out_cand_ids_ptr,  # [max_num_reqs, B+1, K] int64 (indexed by req_state)
    out_cand_probs_ptr,  # [max_num_reqs, B+1, K] float32 (indexed by req_state)
    out_child_rank_ptr,  # [max_num_reqs, B+1] int32 (indexed by req_state)
    out_is_stoch_ptr,  # [max_num_reqs] uint8 (indexed by req_state)
    inv_temp,
    stoch_alpha_base,
    pad_token,
    HAS_NULL: tl.constexpr,
    STOCHASTIC_ENABLED: tl.constexpr,
    K_A: tl.constexpr,
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

    req_temp = 0.0
    req_seed = tl.cast(0, tl.int64)
    base_pos = tl.cast(0, tl.int64)
    is_stoch = False
    if STOCHASTIC_ENABLED:
        req_temp = tl.load(temperature_ptr + req_state).to(tl.float32)
        if req_temp > 0.0:
            is_stoch = True
            req_seed = tl.load(seeds_ptr + req_state).to(tl.int64)
            base_pos = tl.load(sample_pos_ptr + req * N_ROWS).to(tl.int64) - 1
    tl.store(out_is_stoch_ptr + req_state, tl.where(is_stoch, 1, 0).to(tl.uint8))

    f_base = req * MAX_FRONTIER
    # Clear frontier weights to -1.0 (inactive)
    tl.store(f_w_ptr + f_base + offs_f, -1.0)

    # Root node 0 (anchor): parent=-1, depth=0, child_rank=-1, mask[0, 0]=1
    p_base = req_state * B1
    m_base = req_state * B1 * B1
    c_base = req_state * B1 * TOP_K
    tl.store(out_parents_ptr + p_base + 0, -1)
    tl.store(out_depths_ptr + p_base + 0, 0)
    tl.store(out_child_rank_ptr + p_base + 0, -1)
    root_mask = (offs_b1 == 0).to(tl.uint8)
    tl.store(out_mask_ptr + m_base + offs_b1, root_mask, mask=b1_mask)

    # Compute row 0 softmax probabilities: scores[req, 0, 0, 0..K-1]
    s0_base = (req * N_ROWS * TOP_K + 0) * TOP_K
    s0_raw = tl.load(
        scores_ptr + s0_base + offs_k, mask=k_mask, other=float("-inf")
    ).to(tl.float32)
    eff_inv_t0 = (1.0 / req_temp) if is_stoch else inv_temp
    s0 = s0_raw * eff_inv_t0
    s0_max = tl.max(s0, axis=0)
    e0 = tl.exp(s0 - s0_max)
    e0 = tl.where(k_mask, e0, 0.0)
    q0 = e0 / tl.sum(e0, axis=0)
    p0 = q0
    recall0 = 1.0
    if HAS_NULL:
        d0 = (
            tl.load(null_delta_ptr + (req * N_ROWS + 0) * K_A + 0).to(tl.float32)
            + tl.load(null_beta_ptr + 0).to(tl.float32)
        )
        recall0 = 1.0 / (1.0 + tl.exp(d0))
        p0 = p0 * recall0

    # Push top MAX_BR children of root (row 0) into frontier slots 0..MAX_BR-1
    f_size = 0
    if is_stoch:
        c0_ids = tl.load(
            cand_ids_ptr + (req * N_ROWS + 0) * TOP_K + offs_k, mask=k_mask, other=0
        )
        tl.store(out_cand_ids_ptr + c_base + 0 * TOP_K + offs_k, c0_ids, mask=k_mask)
        tl.store(out_cand_probs_ptr + c_base + 0 * TOP_K + offs_k, q0, mask=k_mask)
        # Gumbel-Top-MAX_BR without-replacement sampling from q0, paired with
        # deterministic sorted weights so each child's inclusion is statistically
        # independent of its own sampled token and siblings always enter in exact
        # sampling prefix order. When Null-Candidate probe is available (HAS_NULL),
        # use its calibrated row recall R_u directly as rank-0 acceptance alpha0
        # (zero manual hyperparameters!).
        u0 = murmur3_uniform32(
            req_seed, (base_pos << 8) ^ _DRAFT_NOISE_SALT, offs_k.to(tl.uint32)
        )
        g0 = tl.log(tl.maximum(q0, 1e-30)) - tl.log(-tl.log(tl.maximum(u0, 1e-7)))
        g0 = tl.where(k_mask & (q0 > 1e-12), g0, -1e30)
        rem_g0 = g0
        rem_q0 = q0
        q0_top = tl.max(q0, axis=0)
        use_alpha0 = stoch_alpha_base > 0.0
        alpha0 = (stoch_alpha_base + (1.0 - stoch_alpha_base) * q0_top) * recall0
        rem_tail0 = tl.maximum(1e-6, 1.0 - q0_top)
        for r_idx in range(MAX_BR):
            best_r = tl.argmax(rem_g0, axis=0)
            best_g = tl.max(rem_g0, axis=0)
            det_r = tl.argmax(rem_q0, axis=0)
            det_w = tl.max(rem_q0, axis=0)
            if use_alpha0:
                eff_w = tl.where(
                    r_idx == 0,
                    alpha0,
                    (1.0 - alpha0) * (det_w / rem_tail0),
                )
            else:
                eff_w = det_w
            if (best_g > -1e20) & (eff_w > 0.0):
                tl.store(f_w_ptr + f_base + f_size, eff_w)
                tl.store(f_parent_ptr + f_base + f_size, 0)
                tl.store(f_row_ptr + f_base + f_size, 0)
                tl.store(f_rank_ptr + f_base + f_size, best_r)
                f_size += 1
                rem_g0 = tl.where(offs_k == best_r, -1e30, rem_g0)
                rem_q0 = tl.where(offs_k == det_r, -1.0, rem_q0)
    else:
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
            # Degenerate fallback: attach pad_token to root (child_rank = -1)
            tl.store(out_draft_tokens_ptr + req * BUDGET + (node_id - 1), pad_token)
            tl.store(
                out_node_weights_ptr + req * out_node_weights_stride + (node_id - 1),
                0.0,
            )
            tl.store(out_tree_toks_ptr + req_state * BUDGET + (node_id - 1), pad_token)
            tl.store(out_parents_ptr + p_base + node_id, 0)
            tl.store(out_depths_ptr + p_base + node_id, 1)
            tl.store(out_child_rank_ptr + p_base + node_id, -1)
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
            tl.store(
                out_node_weights_ptr + req * out_node_weights_stride + (node_id - 1),
                best_w,
            )
            tl.store(out_tree_toks_ptr + req_state * BUDGET + (node_id - 1), tok)
            tl.store(out_parents_ptr + p_base + node_id, par_id)
            tl.store(out_depths_ptr + p_base + node_id, row + 1)
            tl.store(out_child_rank_ptr + p_base + node_id, rank)

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
                sn_raw = tl.load(
                    scores_ptr + sn_base + offs_k, mask=k_mask, other=float("-inf")
                ).to(tl.float32)
                eff_inv_tn = (1.0 / req_temp) if is_stoch else inv_temp
                sn = sn_raw * eff_inv_tn
                sn_max = tl.max(sn, axis=0)
                en = tl.exp(sn - sn_max)
                en = tl.where(k_mask, en, 0.0)
                qn = en / tl.sum(en, axis=0)
                pn = qn
                recalln = 1.0
                if HAS_NULL:
                    beta_n = tl.load(null_beta_ptr + next_row).to(tl.float32)
                    if rank < K_A:
                        dn = (
                            tl.load(
                                null_delta_ptr + (req * N_ROWS + next_row) * K_A + rank
                            ).to(tl.float32)
                            + beta_n
                        )
                        recalln = 1.0 / (1.0 + tl.exp(dn))
                        pn = pn * recalln
                    else:
                        recalln = 1.0 / (1.0 + tl.exp(beta_n))
                if is_stoch:
                    cn_ids = tl.load(
                        cand_ids_ptr + (req * N_ROWS + next_row) * TOP_K + offs_k,
                        mask=k_mask,
                        other=0,
                    )
                    tl.store(
                        out_cand_ids_ptr + c_base + node_id * TOP_K + offs_k,
                        cn_ids,
                        mask=k_mask,
                    )
                    tl.store(
                        out_cand_probs_ptr + c_base + node_id * TOP_K + offs_k,
                        qn,
                        mask=k_mask,
                    )
                    un = murmur3_uniform32(
                        req_seed,
                        ((base_pos << 8) ^ node_id) ^ _DRAFT_NOISE_SALT,
                        offs_k.to(tl.uint32),
                    )
                    gn = tl.log(tl.maximum(qn, 1e-30)) - tl.log(
                        -tl.log(tl.maximum(un, 1e-7))
                    )
                    gn = tl.where(k_mask & (qn > 1e-12), gn, -1e30)
                    rem_gn = gn
                    rem_qn = qn
                    qn_top = tl.max(qn, axis=0)
                    use_alphan = stoch_alpha_base > 0.0
                    alphan = (
                        stoch_alpha_base + (1.0 - stoch_alpha_base) * qn_top
                    ) * recalln
                    rem_tailn = tl.maximum(1e-6, 1.0 - qn_top)
                    for r_idx in range(MAX_BR):
                        br = tl.argmax(rem_gn, axis=0)
                        bg = tl.max(rem_gn, axis=0)
                        det_r = tl.argmax(rem_qn, axis=0)
                        det_bw = tl.max(rem_qn, axis=0)
                        if use_alphan:
                            eff_bw = tl.where(
                                r_idx == 0,
                                alphan,
                                (1.0 - alphan) * (det_bw / rem_tailn),
                            )
                        else:
                            eff_bw = det_bw
                        wn = best_w * eff_bw
                        if (bg > -1e20) & (wn > 0.0):
                            tl.store(f_w_ptr + f_base + f_size, wn)
                            tl.store(f_parent_ptr + f_base + f_size, node_id)
                            tl.store(f_row_ptr + f_base + f_size, next_row)
                            tl.store(f_rank_ptr + f_base + f_size, br)
                            f_size += 1
                            rem_gn = tl.where(offs_k == br, -1e30, rem_gn)
                            rem_qn = tl.where(offs_k == det_r, -1.0, rem_qn)
                else:
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
    mask_ptr,  # [num_reqs, MASK_STRIDE, MASK_STRIDE] uint8
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
    MASK_STRIDE: tl.constexpr,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    b = tl.program_id(0)
    h_kv = tl.program_id(1)

    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    q_len = q_end - q_start
    if q_len <= 0:
        return

    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, D)
    q_valid = offs_q < q_len
    tok_idx = q_start + offs_q

    # Load K, V, and tree_mask ONCE per (req, kv_head) and reuse across all G query heads.
    k = tl.load(
        k_ptr + tok_idx[:, None] * k_s0 + h_kv * k_s1 + offs_d[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    v = tl.load(
        v_ptr + tok_idx[:, None] * v_s0 + h_kv * v_s1 + offs_d[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    m_ptrs = (
        mask_ptr + (b * MASK_STRIDE + offs_q[:, None]) * MASK_STRIDE + offs_q[None, :]
    )
    m_2d = tl.load(m_ptrs, mask=q_valid[:, None] & q_valid[None, :], other=0)
    allowed = q_valid[:, None] & q_valid[None, :] & (m_2d != 0)

    pref_len = tl.load(prefix_lens_ptr + b)
    h_base = h_kv * G

    for g in tl.static_range(G):
        h = h_base + g
        q = tl.load(
            q_ptr + tok_idx[:, None] * q_s0 + h * q_s1 + offs_d[None, :],
            mask=q_valid[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        scores = tl.where(allowed, scores, float("-inf"))

        max_s = tl.max(scores, axis=1)
        exp_s = tl.exp(scores - max_s[:, None])
        exp_s = tl.where(allowed, exp_s, 0.0)
        sum_e = tl.sum(exp_s, axis=1)
        lse_t = max_s + tl.log(sum_e)
        probs = (exp_s / sum_e[:, None]).to(v.dtype)

        out_t = tl.dot(probs, v, out_dtype=tl.float32)

        if pref_len > 0:
            out_p = tl.load(
                out_p_ptr + tok_idx[:, None] * op_s0 + h * op_s1 + offs_d[None, :],
                mask=q_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            lse_p = tl.load(
                lse_p_ptr + h * lp_s0 + tok_idx * lp_s1,
                mask=q_valid,
                other=float("-inf"),
            ).to(tl.float32)
            m = tl.maximum(lse_p, lse_t)
            ep = tl.exp(lse_p - m)
            et = tl.exp(lse_t - m)
            out_m = (out_p * ep[:, None] + out_t * et[:, None]) / (ep + et)[:, None]
        else:
            out_m = out_t

        tl.store(
            out_ptr + tok_idx[:, None] * o_s0 + h * o_s1 + offs_d[None, :],
            out_m.to(q_ptr.dtype.element_ty),
            mask=q_valid[:, None],
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
    mask_stride: int | None = None,
) -> None:
    _, H, D = q.shape
    H_kv = k.shape[1]
    ms = mask_stride if mask_stride is not None else tree_mask.shape[1]
    BLOCK_Q = max(16, triton.next_power_of_2(max_q))
    grid = (num_reqs, H_kv)
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
        MASK_STRIDE=ms,
        H=H,
        H_kv=H_kv,
        G=H // H_kv,
        D=D,
        BLOCK_Q=BLOCK_Q,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# 2b. Unified Single-Pass Split-KV Paged-KV Tree Attention (Triton)
# ---------------------------------------------------------------------------


@triton.jit
def _single_pass_tree_attn_kernel(
    Q_ptr,  # [total_q, H_q, D]
    K_cache_ptr,  # [num_blocks, page_size, H_kv, D]
    V_cache_ptr,  # [num_blocks, page_size, H_kv, D]
    Out_ptr,  # [total_q, H_q, D] (used when NUM_SPLITS == 1)
    Split_o_ptr,  # [total_q, H_q, NUM_SPLITS, D] fp32 (when NUM_SPLITS > 1)
    Split_lse_ptr,  # [total_q, H_q, NUM_SPLITS] fp32 (when NUM_SPLITS > 1)
    Block_table_ptr,  # [B, max_pages] int32
    Cu_q_ptr,  # [B + 1] int32
    Seq_k_ptr,  # [B] int32 (prefix_len + q_len)
    Bitmasks_ptr,  # [B, 32] int32 (packed uint32 ancestor bitmask per row)
    sm_scale: tl.constexpr,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_ot,
    stride_oh,
    stride_sot,
    stride_soh,
    stride_sos,
    stride_slt,
    stride_slh,
    stride_bt_b,
    stride_bm_b,
    G: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    h_kv = tl.program_id(1)
    pid_ms = tl.program_id(2)
    pid_m = pid_ms // NUM_SPLITS
    split_id = pid_ms % NUM_SPLITS

    q_start = tl.load(Cu_q_ptr + b)
    q_end = tl.load(Cu_q_ptr + b + 1)
    q_len = q_end - q_start
    if (q_len <= 0) | (pid_m * BLOCK_M >= q_len * G):
        return

    seq_k = tl.load(Seq_k_ptr + b)
    prefix_len = tl.maximum(seq_k - q_len, 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_local = offs_m // G
    g = offs_m % G
    h_q = h_kv * G + g
    q_valid = q_local < q_len

    total_n_blocks = (seq_k + BLOCK_N - 1) // BLOCK_N
    if NUM_SPLITS > 1:
        blocks_per_split = (total_n_blocks + NUM_SPLITS - 1) // NUM_SPLITS
        blk_start = split_id * blocks_per_split
        blk_end = tl.minimum(blk_start + blocks_per_split, total_n_blocks)
        if blk_start >= blk_end:
            lse_ptrs = (
                Split_lse_ptr
                + (q_start + q_local) * stride_slt
                + h_q * stride_slh
                + split_id
            )
            tl.store(lse_ptrs, float("-inf"), mask=q_valid)
            return
    else:
        blk_start = 0
        blk_end = total_n_blocks

    offs_d = tl.arange(0, HEAD_DIM)
    safe_q_local = tl.minimum(q_local, tl.maximum(q_len - 1, 0))
    q_ptrs = (
        Q_ptr
        + (q_start + safe_q_local)[:, None] * stride_qt
        + h_q[:, None] * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)

    row_bm = tl.load(
        Bitmasks_ptr + b * stride_bm_b + tl.minimum(safe_q_local, 31),
        mask=q_valid,
        other=0,
    ).to(tl.uint32)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    sm_scale_log2: tl.constexpr = sm_scale * 1.4426950408889634
    offs_n = tl.arange(0, BLOCK_N)
    bt_base = Block_table_ptr + b * stride_bt_b
    k_base = K_cache_ptr + h_kv * stride_kh
    v_base = V_cache_ptr + h_kv * stride_vh

    # Loop 1: Full unmasked prefix blocks [blk_start, min(blk_end, prefix_full_blocks))
    # Coalesced [BLOCK_N, HEAD_DIM] vector loads on both K and V (stride-1 along HEAD_DIM).
    prefix_full_blocks = tl.maximum(prefix_len, 0) // BLOCK_N
    unmasked_end = tl.minimum(blk_end, prefix_full_blocks)
    for blk in range(blk_start, unmasked_end):
        pos_k = blk * BLOCK_N + offs_n
        page_idx = tl.load(bt_base + (pos_k // PAGE_SIZE)).to(tl.int64)
        page_off = (pos_k % PAGE_SIZE).to(tl.int64)

        k_ptrs = (
            k_base + page_idx[:, None] * stride_kb + page_off[:, None] * stride_kp + offs_d[None, :]
        )
        k = tl.load(k_ptrs)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = (
            v_base + page_idx[:, None] * stride_vb + page_off[:, None] * stride_vp + offs_d[None, :]
        )
        v = tl.load(v_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij

    # Loop 2: Tail prefix + intra-tree bitmask blocks [max(blk_start, prefix_full_blocks), blk_end)
    masked_start = tl.maximum(blk_start, prefix_full_blocks)
    for blk in range(masked_start, blk_end):
        pos_k = blk * BLOCK_N + offs_n
        kv_valid = pos_k < seq_k
        safe_pos_k = tl.minimum(pos_k, tl.maximum(seq_k - 1, 0))
        page_idx = tl.load(
            bt_base + (safe_pos_k // PAGE_SIZE), mask=kv_valid, other=0
        ).to(tl.int64)
        page_off = (safe_pos_k % PAGE_SIZE).to(tl.int64)

        k_ptrs = (
            k_base + page_idx[:, None] * stride_kb + page_off[:, None] * stride_kp + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=kv_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2

        rel_k = pos_k[None, :] - prefix_len
        in_prefix = rel_k < 0
        shift = tl.maximum(tl.minimum(rel_k, 31), 0).to(tl.uint32)
        in_tree = (rel_k < q_len) & (((row_bm[:, None] >> shift) & 1) != 0)
        keep = kv_valid[None, :] & (in_prefix | in_tree)
        qk = tl.where(keep, qk, float("-inf"))

        row_max = tl.max(qk, 1)
        m_ij = tl.maximum(m_i, row_max)
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])
        safe_m_i = tl.where(m_i == float("-inf"), 0.0, m_i)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(safe_m_i - safe_m_ij))
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = (
            v_base + page_idx[:, None] * stride_vb + page_off[:, None] * stride_vp + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij

    out_val = acc / tl.maximum(l_i[:, None], 1e-20)
    if NUM_SPLITS == 1:
        out_ptrs = (
            Out_ptr
            + (q_start + q_local)[:, None] * stride_ot
            + h_q[:, None] * stride_oh
            + offs_d[None, :]
        )
        tl.store(out_ptrs, out_val.to(Out_ptr.dtype.element_ty), mask=q_valid[:, None])
    else:
        has_mass = (m_i > float("-inf")) & (l_i > 0.0)
        lse_val = tl.where(
            has_mass, m_i + tl.math.log2(tl.maximum(l_i, 1e-20)), float("-inf")
        )
        so_ptrs = (
            Split_o_ptr
            + (q_start + q_local)[:, None] * stride_sot
            + h_q[:, None] * stride_soh
            + split_id * stride_sos
            + offs_d[None, :]
        )
        tl.store(so_ptrs, out_val, mask=q_valid[:, None])
        sl_ptrs = (
            Split_lse_ptr
            + (q_start + q_local) * stride_slt
            + h_q * stride_slh
            + split_id
        )
        tl.store(sl_ptrs, lse_val, mask=q_valid)


@triton.jit
def _single_pass_tree_reduce_kernel(
    Split_o_ptr,  # [total_q, H_q, NUM_SPLITS, D] fp32
    Split_lse_ptr,  # [total_q, H_q, NUM_SPLITS] fp32
    Out_ptr,  # [total_q, H_q, D] bf16/fp16
    stride_sot,
    stride_soh,
    stride_sos,
    stride_slt,
    stride_slh,
    stride_ot,
    stride_oh,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    tok = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    s_mask = offs_s < NUM_SPLITS
    lse = tl.load(
        Split_lse_ptr + tok * stride_slt + h * stride_slh + offs_s,
        mask=s_mask,
        other=float("-inf"),
    )
    m_max = tl.max(lse, axis=0)
    safe_m = tl.where(m_max == float("-inf"), 0.0, m_max)
    w = tl.where(s_mask & (lse > float("-inf")), tl.math.exp2(lse - safe_m), 0.0)
    denom = tl.maximum(tl.sum(w, axis=0), 1e-20)
    w_norm = w / denom

    offs_d = tl.arange(0, HEAD_DIM)
    so_base = Split_o_ptr + tok * stride_sot + h * stride_soh
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        ws = tl.sum(tl.where(offs_s == s, w_norm, 0.0), axis=0)
        if ws > 0.0:
            sv = tl.load(so_base + s * stride_sos + offs_d)
            acc += ws * sv

    tl.store(
        Out_ptr + tok * stride_ot + h * stride_oh + offs_d,
        acc.to(Out_ptr.dtype.element_ty),
    )


def single_pass_tree_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    tree_bitmask: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: float,
    tree_state: CRLTreeState | None = None,
) -> None:
    """Unified single-pass Split-KV Paged-KV + uint32 Tree Bitmask attention."""
    total_q, H_q, D = q.shape
    page_size = key_cache.shape[1]
    H_kv = key_cache.shape[2]
    G = H_q // H_kv
    num_reqs = seqused_k.shape[0]
    if total_q == 0 or num_reqs == 0:
        return

    # Deterministic per-(num_reqs, H_kv) split count (100% FULL CUDA Graph safe)
    total_ctas = num_reqs * H_kv
    if total_ctas >= 64:
        num_splits = 1
    elif total_ctas >= 16:
        num_splits = 4
    else:
        num_splits = 8

    mq_g = max_seqlen_q * G
    if mq_g >= 48:
        block_m = 64
    elif mq_g >= 17:
        block_m = 32
    else:
        block_m = 16
    n_warps = 4
    block_n = 64
    num_m_tiles = triton.cdiv(mq_g, block_m)

    if num_splits > 1:
        if tree_state is not None:
            split_o, split_lse = tree_state.get_split_bufs(
                total_q, H_q, num_splits, D, q.device
            )
        else:
            split_o = torch.empty(
                (total_q, H_q, num_splits, D), dtype=torch.float32, device=q.device
            )
            split_lse = torch.empty(
                (total_q, H_q, num_splits), dtype=torch.float32, device=q.device
            )
    else:
        # Dummy pointers (unused when NUM_SPLITS == 1)
        split_o = out
        split_lse = seqused_k

    grid = (num_reqs, H_kv, num_m_tiles * num_splits)
    _single_pass_tree_attn_kernel[grid](
        q,
        key_cache,
        value_cache,
        out,
        split_o,
        split_lse,
        block_table,
        cu_seqlens_q,
        seqused_k,
        tree_bitmask,
        sm_scale=sm_scale,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_kb=key_cache.stride(0),
        stride_kp=key_cache.stride(1),
        stride_kh=key_cache.stride(2),
        stride_vb=value_cache.stride(0),
        stride_vp=value_cache.stride(1),
        stride_vh=value_cache.stride(2),
        stride_ot=out.stride(0),
        stride_oh=out.stride(1),
        stride_sot=split_o.stride(0),
        stride_soh=split_o.stride(1) if split_o.ndim >= 3 else 1,
        stride_sos=split_o.stride(2) if split_o.ndim >= 4 else 1,
        stride_slt=split_lse.stride(0),
        stride_slh=split_lse.stride(1) if split_lse.ndim >= 2 else 1,
        stride_bt_b=block_table.stride(0),
        stride_bm_b=tree_bitmask.stride(0),
        G=G,
        HEAD_DIM=D,
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        num_warps=n_warps,
        num_stages=3,
    )

    if num_splits > 1:
        _single_pass_tree_reduce_kernel[(total_q, H_q)](
            split_o,
            split_lse,
            out,
            stride_sot=split_o.stride(0),
            stride_soh=split_o.stride(1),
            stride_sos=split_o.stride(2),
            stride_slt=split_lse.stride(0),
            stride_slh=split_lse.stride(1),
            stride_ot=out.stride(0),
            stride_oh=out.stride(1),
            HEAD_DIM=D,
            NUM_SPLITS=num_splits,
            BLOCK_S=triton.next_power_of_2(num_splits),
            num_warps=2,
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
    step_tree_bitmask_ptr,  # [max_num_reqs, 32] int32 (out, packed uint32 per row)
    B1: tl.constexpr,
    BLOCK_B1: tl.constexpr,
):
    r = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + r)
    pref_len = tl.load(num_computed_ptr + req_state)
    tl.store(step_prefix_lens_ptr + r, pref_len)

    if tl.load(is_tree_req_ptr + r) == 0:
        tl.store(step_tree_mask_ptr + r * B1 * B1 + 0, 1)
        tl.store(step_tree_bitmask_ptr + r * 32 + 0, 1)
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

    # Copy tree_mask[req_state] -> step_tree_mask[r] and pack uint32 row bitmasks
    src_m = tree_mask_ptr + req_state * B1 * B1
    dst_m = step_tree_mask_ptr + r * B1 * B1
    b1_mask = offs < B1
    bit_weights = tl.cast(1, tl.uint32) << tl.minimum(offs, 31).to(tl.uint32)
    for row in range(B1):
        vals = tl.load(src_m + row * B1 + offs, mask=b1_mask, other=0)
        tl.store(dst_m + row * B1 + offs, vals, mask=b1_mask)
        if B1 <= 32:
            bits = tl.sum(
                tl.where(b1_mask & (vals != 0), bit_weights, tl.cast(0, tl.uint32)),
                axis=0,
            )
            tl.store(step_tree_bitmask_ptr + r * 32 + row, bits.to(tl.int32))


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


@triton.jit
def _vocab_argmax(row_ptr, vocab_size, BLOCK_V: tl.constexpr):
    best_val = float("-inf")
    best_idx = tl.cast(0, tl.int64)
    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(vals, axis=0)
        blk_arg = tl.argmax(vals, axis=0).to(tl.int64) + start
        if blk_max > best_val:
            best_val = blk_max
            best_idx = blk_arg
    return best_idx


@triton.jit
def _vocab_lse(row_ptr, vocab_size, BLOCK_V: tl.constexpr):
    max_val = float("-inf")
    sum_exp = 0.0
    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(vals, axis=0)
        new_max = tl.maximum(max_val, blk_max)
        scale_old = tl.where(max_val > float("-inf"), tl.exp(max_val - new_max), 0.0)
        blk_sum = tl.sum(
            tl.where(mask & (vals > float("-inf")), tl.exp(vals - new_max), 0.0),
            axis=0,
        )
        sum_exp = sum_exp * scale_old + blk_sum
        max_val = new_max
    return max_val + tl.log(tl.maximum(sum_exp, 1e-30))


@triton.jit
def _vocab_gumbel_sample(
    row_ptr,
    vocab_size,
    seed,
    pos_key,
    cand_ids_ptr,
    EXCLUDE_CANDS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    best_g = float("-inf")
    best_idx = tl.cast(0, tl.int64)
    for start in range(0, vocab_size, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        if EXCLUDE_CANDS:
            for k in tl.static_range(TOP_K):
                ck = tl.load(cand_ids_ptr + k)
                vals = tl.where(offs == ck, float("-inf"), vals)
        u = murmur3_uniform32(seed, pos_key, offs.to(tl.uint32))
        g = vals - tl.log(-tl.log(tl.maximum(u, 1e-7)))
        g = tl.where(mask & (vals > float("-inf")), g, float("-inf"))
        blk_max = tl.max(g, axis=0)
        blk_arg = tl.argmax(g, axis=0).to(tl.int64) + start
        if blk_max > best_g:
            best_g = blk_max
            best_idx = blk_arg
    return best_idx


@triton.jit
def _fused_row_lse_split_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    split_size,
    part_max_ptr,  # [total_logits, NUM_SPLITS] float32
    part_sum_ptr,  # [total_logits, NUM_SPLITS] float32
    NUM_SPLITS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    sid = tl.program_id(1)
    v_start = sid * split_size
    v_end = tl.minimum(v_start + split_size, vocab_size)
    row_ptr = logits_ptr + tl.cast(row, tl.int64) * logits_stride

    max_val = float("-inf")
    sum_exp = 0.0
    n_elems = tl.maximum(0, v_end - v_start)
    num_iters = (n_elems + BLOCK_V - 1) // BLOCK_V
    for it in range(num_iters):
        start = v_start + it * BLOCK_V
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < v_end
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(vals, axis=0)
        new_max = tl.maximum(max_val, blk_max)
        scale_old = tl.where(max_val > float("-inf"), tl.exp(max_val - new_max), 0.0)
        blk_sum = tl.sum(
            tl.where(mask & (vals > float("-inf")), tl.exp(vals - new_max), 0.0),
            axis=0,
        )
        sum_exp = sum_exp * scale_old + blk_sum
        max_val = new_max

    out_idx = row * NUM_SPLITS + sid
    tl.store(part_max_ptr + out_idx, max_val)
    tl.store(part_sum_ptr + out_idx, sum_exp)


@triton.jit
def _reduce_row_lse_kernel(
    part_max_ptr,  # [total_logits, NUM_SPLITS] float32
    part_sum_ptr,  # [total_logits, NUM_SPLITS] float32
    row_lse_ptr,  # [total_logits] float32
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_S)
    mask = offs < NUM_SPLITS
    base = row * NUM_SPLITS
    pmax = tl.load(part_max_ptr + base + offs, mask=mask, other=float("-inf"))
    psum = tl.load(part_sum_ptr + base + offs, mask=mask, other=0.0)
    gmax = tl.max(pmax, axis=0)
    scaled = tl.where(mask & (pmax > float("-inf")), psum * tl.exp(pmax - gmax), 0.0)
    gsum = tl.sum(scaled, axis=0)
    lse = gmax + tl.log(tl.maximum(gsum, 1e-30))
    tl.store(row_lse_ptr + row, lse)


@triton.jit
def _parallel_vocab_bonus_split_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    split_size,
    idx_mapping_ptr,  # [num_reqs] int64
    positions_ptr,  # [total_logits] int64
    seeds_ptr,  # [max_num_reqs] int64
    tree_cand_ids_ptr,  # [max_num_reqs, B+1, K] int64
    stop_row_ptr,  # [num_reqs] int32
    stop_u_ptr,  # [num_reqs] int32
    stop_mode_ptr,  # [num_reqs] int32 (0=done, 1=full gumbel, 2=excl-cands gumbel, 3=argmax)
    part_val_ptr,  # [num_reqs, BONUS_SPLITS] float32
    part_idx_ptr,  # [num_reqs, BONUS_SPLITS] int64
    B1: tl.constexpr,
    TOP_K: tl.constexpr,
    BONUS_SPLITS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    r = tl.program_id(0)
    mode = tl.load(stop_mode_ptr + r)
    if mode == 0:
        return

    sid = tl.program_id(1)
    row = tl.load(stop_row_ptr + r)
    req_state = tl.load(idx_mapping_ptr + r)
    seed = tl.load(seeds_ptr + req_state).to(tl.int64)
    pos_u = tl.load(positions_ptr + row).to(tl.int64)
    stop_u = tl.load(stop_u_ptr + r)

    v_start = sid * split_size
    v_end = tl.minimum(v_start + split_size, vocab_size)
    row_ptr = logits_ptr + tl.cast(row, tl.int64) * logits_stride
    cand_u_ptr = tree_cand_ids_ptr + (req_state * B1 + stop_u) * TOP_K

    best_val = float("-inf")
    best_idx = tl.cast(0, tl.int64)
    n_elems = tl.maximum(0, v_end - v_start)
    num_iters = (n_elems + BLOCK_V - 1) // BLOCK_V
    for it in range(num_iters):
        start = v_start + it * BLOCK_V
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < v_end
        vals = tl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
        if mode == 2:
            for k in tl.static_range(TOP_K):
                ck = tl.load(cand_u_ptr + k)
                vals = tl.where(offs == ck, float("-inf"), vals)
        if mode == 3:
            scores = tl.where(mask, vals, float("-inf"))
        else:
            u = murmur3_uniform32(seed, pos_u, offs.to(tl.uint32))
            scores = vals - tl.log(-tl.log(tl.maximum(u, 1e-7)))
            scores = tl.where(mask & (vals > float("-inf")), scores, float("-inf"))
        blk_max = tl.max(scores, axis=0)
        blk_arg = tl.argmax(scores, axis=0).to(tl.int64) + start
        if blk_max > best_val:
            best_val = blk_max
            best_idx = blk_arg

    out_idx = r * BONUS_SPLITS + sid
    tl.store(part_val_ptr + out_idx, best_val)
    tl.store(part_idx_ptr + out_idx, best_idx)


@triton.jit
def _reduce_vocab_bonus_kernel(
    stop_mode_ptr,  # [num_reqs] int32
    stop_step_ptr,  # [num_reqs] int32
    part_val_ptr,  # [num_reqs, BONUS_SPLITS] float32
    part_idx_ptr,  # [num_reqs, BONUS_SPLITS] int64
    out_sampled_ptr,  # [num_reqs, B+1] int64
    B1: tl.constexpr,
    BONUS_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    r = tl.program_id(0)
    if tl.load(stop_mode_ptr + r) == 0:
        return
    offs = tl.arange(0, BLOCK_S)
    mask = offs < BONUS_SPLITS
    base = r * BONUS_SPLITS
    vals = tl.load(part_val_ptr + base + offs, mask=mask, other=float("-inf"))
    idxs = tl.load(part_idx_ptr + base + offs, mask=mask, other=0)
    best_s = tl.argmax(vals, axis=0)
    best_tok = tl.sum(tl.where(offs == best_s, idxs, 0), axis=0)
    step = tl.load(stop_step_ptr + r)
    tl.store(out_sampled_ptr + r * B1 + step, best_tok)


@triton.jit
def _tree_verify_stochastic_kernel(
    logits_ptr,  # [total_logits, V] (already processed by apply_sampling_params)
    row_lse_ptr,  # [total_logits] float32 (precomputed in parallel across all SMs!)
    logits_stride,
    vocab_size,
    idx_mapping_ptr,  # [num_reqs] int64
    cu_num_logits_ptr,  # [num_reqs + 1] int32
    positions_ptr,  # [total_logits] int64
    seeds_ptr,  # [max_num_reqs] int64
    temperature_ptr,  # [max_num_reqs] float32
    tree_tokens_ptr,  # [max_num_reqs, B] int64
    tree_parents_ptr,  # [max_num_reqs, B+1] int32
    tree_cand_ids_ptr,  # [max_num_reqs, B+1, K] int64
    tree_cand_probs_ptr,  # [max_num_reqs, B+1, K] float32
    tree_child_rank_ptr,  # [max_num_reqs, B+1] int32
    tree_is_stoch_ptr,  # [max_num_reqs] uint8
    out_sampled_ptr,  # [num_reqs, B+1] int64 (-1 padded)
    out_acc_nodes_ptr,  # [num_reqs, B+1] int32
    out_num_sampled_ptr,  # [num_reqs] int32
    out_num_rejected_ptr,  # [num_reqs] int32
    stop_row_ptr,  # [num_reqs] int32
    stop_step_ptr,  # [num_reqs] int32
    stop_u_ptr,  # [num_reqs] int32
    stop_mode_ptr,  # [num_reqs] int32 (0=done, 1=full gumbel, 2=excl-cands gumbel, 3=argmax)
    BUDGET: tl.constexpr,
    B1: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    r = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + r)
    l_start = tl.load(cu_num_logits_ptr + r)
    l_end = tl.load(cu_num_logits_ptr + r + 1)
    n_logits = l_end - l_start

    tl.store(stop_mode_ptr + r, 0)
    for j in range(B1):
        tl.store(out_sampled_ptr + r * B1 + j, -1)
        tl.store(out_acc_nodes_ptr + r * B1 + j, 0)

    if n_logits <= 0:
        tl.store(out_num_sampled_ptr + r, 0)
        tl.store(out_num_rejected_ptr + r, 0)
        return

    temp = tl.load(temperature_ptr + req_state).to(tl.float32)
    seed = tl.load(seeds_ptr + req_state).to(tl.int64)
    is_stoch = tl.load(tree_is_stoch_ptr + req_state) != 0
    p_base = req_state * B1
    t_base = req_state * BUDGET
    c_base = req_state * B1 * TOP_K

    # Single-logit (prefill-only) request: offload to parallel vocab bonus kernel
    if n_logits == 1:
        tl.store(out_acc_nodes_ptr + r * B1 + 0, 0)
        tl.store(out_num_sampled_ptr + r, 1)
        tl.store(out_num_rejected_ptr + r, 0)
        tl.store(stop_row_ptr + r, l_start)
        tl.store(stop_step_ptr + r, 0)
        tl.store(stop_u_ptr + r, 0)
        tl.store(stop_mode_ptr + r, 3 if temp == 0.0 else 1)
        return

    # Greedy request inside a mixed batch (or deterministic tree fallback)
    if (temp == 0.0) | (not is_stoch):
        u = 0
        step = 0
        keep_going = True
        for _ in range(B1):
            if keep_going:
                row_ptr = logits_ptr + tl.cast(l_start + u, tl.int64) * logits_stride
                if temp == 0.0:
                    t_star = _vocab_argmax(row_ptr, vocab_size, BLOCK_V)
                else:
                    pos_u = tl.load(positions_ptr + l_start + u).to(tl.int64)
                    t_star = _vocab_gumbel_sample(
                        row_ptr,
                        vocab_size,
                        seed,
                        pos_u,
                        tree_cand_ids_ptr,
                        False,
                        TOP_K,
                        BLOCK_V,
                    )
                tl.store(out_sampled_ptr + r * B1 + step, t_star)
                tl.store(out_acc_nodes_ptr + r * B1 + step, u)
                step += 1
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
        return

    # Stochastic without-replacement recursive residual tree rejection sampling
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < TOP_K
    u = 0
    step = 0
    keep_going = True

    for _ in range(B1):
        if keep_going:
            tl.store(out_acc_nodes_ptr + r * B1 + step, u)
            row_ptr = logits_ptr + tl.cast(l_start + u, tl.int64) * logits_stride
            pos_u = tl.load(positions_ptr + l_start + u).to(tl.int64)

            # Check if u has any valid children in 1..n_logits-1
            has_child = False
            for v in range(1, B1):
                if v < n_logits:
                    pv = tl.load(tree_parents_ptr + p_base + v)
                    cr = tl.load(tree_child_rank_ptr + p_base + v)
                    if (pv == u) & (cr >= 0):
                        has_child = True

            if not has_child:
                # Leaf node: offload full-vocab bonus sample to parallel kernel
                tl.store(stop_row_ptr + r, l_start + u)
                tl.store(stop_step_ptr + r, step)
                tl.store(stop_u_ptr + r, u)
                tl.store(stop_mode_ptr + r, 1)
                step += 1
                keep_going = False
            else:
                # Read precomputed parallel target LSE and sparse-support p_cand[16]
                lse_u = tl.load(row_lse_ptr + l_start + u).to(tl.float32)
                cand_u_ptr = tree_cand_ids_ptr + c_base + u * TOP_K
                c_ids = tl.load(cand_u_ptr + offs_k, mask=k_mask, other=0)
                q_orig = tl.load(
                    tree_cand_probs_ptr + c_base + u * TOP_K + offs_k,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                c_logits = tl.load(
                    row_ptr + c_ids,
                    mask=k_mask & (c_ids >= 0) & (c_ids < vocab_size),
                    other=float("-inf"),
                ).to(tl.float32)
                p_cand = tl.where(
                    k_mask & (c_logits > float("-inf")), tl.exp(c_logits - lse_u), 0.0
                )

                p_cur = p_cand
                p_rest = tl.maximum(0.0, 1.0 - tl.sum(p_cand, axis=0))
                rem_S = 1.0
                used = tl.zeros([BLOCK_K], dtype=tl.int1)
                accepted_v = -1

                # Children appear in exact without-replacement sampling order
                for v in range(1, B1):
                    if (v < n_logits) & (accepted_v < 0):
                        pv = tl.load(tree_parents_ptr + p_base + v)
                        cr = tl.load(tree_child_rank_ptr + p_base + v)
                        if (pv == u) & (cr >= 0):
                            is_cr = offs_k == cr
                            q_raw_r = tl.sum(tl.where(is_cr, q_orig, 0.0), axis=0)
                            p_cur_r = tl.sum(tl.where(is_cr, p_cur, 0.0), axis=0)
                            if (rem_S > 1e-12) & (q_raw_r > 1e-12):
                                q_r = q_raw_r / rem_S
                                alpha = tl.minimum(1.0, p_cur_r / q_r)
                                eta = murmur3_uniform32(
                                    seed,
                                    (pos_u << 8) ^ 0x5A5A,
                                    tl.cast(v, tl.uint32),
                                )
                                if eta < alpha:
                                    accepted_v = v
                                    tok_v = tl.load(
                                        tree_tokens_ptr + t_base + (v - 1)
                                    )
                                    tl.store(out_sampled_ptr + r * B1 + step, tok_v)
                                    step += 1
                                else:
                                    q_cur = tl.where(
                                        k_mask & (~used), q_orig / rem_S, 0.0
                                    )
                                    p_new = tl.maximum(0.0, p_cur - q_cur)
                                    Z = tl.sum(p_new, axis=0) + p_rest
                                    if Z > 1e-12:
                                        inv_Z = 1.0 / Z
                                        p_cur = p_new * inv_Z
                                        p_rest = p_rest * inv_Z
                                    else:
                                        p_cur = p_cand
                                        p_rest = tl.maximum(
                                            0.0, 1.0 - tl.sum(p_cand, axis=0)
                                        )
                                    used = used | is_cr
                                    rem_S = tl.maximum(0.0, rem_S - q_raw_r)

                if accepted_v >= 0:
                    u = accepted_v
                else:
                    # All children rejected: exact 2-way mixture sample from residual
                    m_cand = tl.sum(p_cur, axis=0)
                    u_mix = murmur3_uniform32(
                        seed, (pos_u << 8) ^ 0xA5A5, tl.cast(u, tl.uint32)
                    )
                    if (m_cand > 1e-12) & ((u_mix < m_cand) | (p_rest <= 1e-12)):
                        # Sample directly from 16-element residual p_cur in registers!
                        u_c = murmur3_uniform32(
                            seed, pos_u, offs_k.to(tl.uint32)
                        )
                        g_c = tl.log(tl.maximum(p_cur, 1e-30)) - tl.log(
                            -tl.log(tl.maximum(u_c, 1e-7))
                        )
                        g_c = tl.where(k_mask & (p_cur > 1e-12), g_c, float("-inf"))
                        best_k = tl.argmax(g_c, axis=0)
                        bonus = tl.sum(tl.where(offs_k == best_k, c_ids, 0), axis=0)
                        tl.store(out_sampled_ptr + r * B1 + step, bonus)
                    else:
                        # Offload vocab \ C_u residual sample to parallel bonus kernel
                        tl.store(stop_row_ptr + r, l_start + u)
                        tl.store(stop_step_ptr + r, step)
                        tl.store(stop_u_ptr + r, u)
                        tl.store(stop_mode_ptr + r, 2)
                    step += 1
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
        # Persistent stochastic-tree proposal buffers (CUDA-Graph safe, indexed by req_state)
        self.tree_cand_ids = torch.zeros(
            max_num_reqs, self.b1, top_k, dtype=torch.int64, device=device
        )
        self.tree_cand_probs = torch.zeros(
            max_num_reqs, self.b1, top_k, dtype=torch.float32, device=device
        )
        self.tree_child_rank = torch.full(
            (max_num_reqs, self.b1), -1, dtype=torch.int32, device=device
        )
        self.tree_is_stochastic = torch.zeros(
            max_num_reqs, dtype=torch.uint8, device=device
        )
        self.stochastic_enabled = get_crl_tree_stochastic_verify()
        self.stoch_alpha_base = get_crl_tree_stoch_alpha_base()
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
        self.step_node_weights = torch.zeros(
            max_num_reqs, budget, dtype=torch.float32, device=device
        )
        self.step_is_tree_req = torch.ones(
            max_num_reqs, dtype=torch.uint8, device=device
        )
        self._is_tree_req_pinned = torch.ones(
            max_num_reqs, dtype=torch.uint8, pin_memory=True
        )
        self.step_tree_mask = torch.zeros(
            max_num_reqs, self.b1, self.b1, dtype=torch.uint8, device=device
        )
        self.step_tree_bitmask = torch.zeros(
            max_num_reqs, 32, dtype=torch.int32, device=device
        )
        for i in range(self.b1):
            self.step_tree_mask[:, i, : i + 1] = 1
            if i < 32:
                bm_u32 = (1 << (i + 1)) - 1
                bm_i32 = bm_u32 - (1 << 32) if bm_u32 >= (1 << 31) else bm_u32
                self.step_tree_bitmask[:, i] = bm_i32

        # Verification output buffers
        self.step_acc_nodes = torch.zeros(
            max_num_reqs, self.b1, dtype=torch.int32, device=device
        )
        # Persistent scratch buffers for all-SM fused stochastic verification
        self._stop_row = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._stop_step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._stop_u = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._stop_mode = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._bonus_part_val = torch.empty(
            max_num_reqs, 32, dtype=torch.float32, device=device
        )
        self._bonus_part_idx = torch.empty(
            max_num_reqs, 32, dtype=torch.int64, device=device
        )
        self._lse_buf_cap = 0
        self._row_lse: torch.Tensor | None = None
        self._lse_part_max: torch.Tensor | None = None
        self._lse_part_sum: torch.Tensor | None = None
        # Pre-allocated prefix attention output buffer (lazy-sized on first use)
        self._prefix_out_buf: torch.Tensor | None = None
        # Pre-allocated Split-KV scratch buffers for single-pass tree attention
        # (fixed addresses for FULL CUDA Graph capture & replay; max 8 splits * 256 tok)
        self._split_o_buf: torch.Tensor | None = None
        self._split_lse_buf: torch.Tensor | None = None
        # Cached device pointer table for single-launch batched 36-layer KV compaction
        self._kv_ptrs_cache: tuple[int, torch.Tensor] | None = None

    def get_split_bufs(
        self,
        total_q: int,
        num_heads: int,
        num_splits: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-allocate generous capacity (at least 256 tokens * 8 splits) on first
        # call so subsequent CUDA Graph buckets reuse the same fixed addresses.
        cap_q = max(total_q, 256)
        cap_s = max(num_splits, 8)
        if (
            self._split_o_buf is None
            or self._split_o_buf.shape[0] < total_q
            or self._split_o_buf.shape[1] != num_heads
            or self._split_o_buf.shape[2] < num_splits
            or self._split_o_buf.shape[3] != head_dim
        ):
            self._split_o_buf = torch.empty(
                (cap_q, num_heads, cap_s, head_dim), dtype=torch.float32, device=device
            )
            self._split_lse_buf = torch.empty(
                (cap_q, num_heads, cap_s), dtype=torch.float32, device=device
            )
        assert self._split_lse_buf is not None
        return (
            self._split_o_buf[:total_q, :, :num_splits, :],
            self._split_lse_buf[:total_q, :, :num_splits],
        )

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
        null_delta: torch.Tensor | None = None,  # [max_num_reqs, N, K_a]
        null_beta: torch.Tensor | None = None,  # [N]
        out_node_weights: torch.Tensor | None = None,  # [max_num_reqs, B]
        temperature: torch.Tensor | None = None,  # [max_num_reqs] float32
        seeds: torch.Tensor | None = None,  # [max_num_reqs] int64
        sample_positions: torch.Tensor | None = None,  # [num_reqs * N] int64
    ) -> None:
        BLOCK_K = triton.next_power_of_2(self.top_k)
        BLOCK_B1 = triton.next_power_of_2(self.b1)
        has_null = null_delta is not None and null_beta is not None
        use_stoch = (
            self.stochastic_enabled
            and temperature is not None
            and seeds is not None
            and sample_positions is not None
        )
        k_a = int(null_delta.shape[-1]) if has_null else 1
        w_out = (
            out_node_weights
            if out_node_weights is not None
            else self.step_node_weights
        )
        inv_t = 1.0 if has_null else (1.0 / self.temp)
        _build_crl_tree_kernel[(num_reqs,)](
            scores,
            candidate_ids,
            sample_idx_mapping,
            null_delta if has_null else self._f_w,
            null_beta if has_null else self._f_w,
            temperature if use_stoch else self._f_w,
            seeds if use_stoch else self.tree_tokens,
            sample_positions if use_stoch else self.tree_tokens,
            self._f_w,
            self._f_parent,
            self._f_row,
            self._f_rank,
            out_draft_tokens,
            w_out,
            w_out.stride(0),
            self.tree_parents,
            self.tree_depths,
            self.tree_mask,
            self.tree_tokens,
            self.tree_cand_ids,
            self.tree_cand_probs,
            self.tree_child_rank,
            self.tree_is_stochastic,
            inv_t,
            self.stoch_alpha_base,
            pad_token,
            HAS_NULL=has_null,
            STOCHASTIC_ENABLED=use_stoch,
            K_A=k_a,
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
            self._is_tree_req_pinned[:num_reqs].copy_(
                torch.from_numpy(is_tree_req_np)
            )
            self.step_is_tree_req[:num_reqs].copy_(
                self._is_tree_req_pinned[:num_reqs], non_blocking=True
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
            self.step_tree_bitmask,
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

    def verify_trees_stochastic(
        self,
        processed_logits: torch.Tensor,  # [total_logits, V]
        idx_mapping: torch.Tensor,  # [num_reqs] int64
        cu_num_logits: torch.Tensor,  # [num_reqs + 1] int32
        positions: torch.Tensor,  # [total_logits] int64
        seeds: torch.Tensor,  # [max_num_reqs] int64
        temperature: torch.Tensor,  # [max_num_reqs] float32
        num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_sampled = torch.empty(
            (num_reqs, self.b1), dtype=torch.int64, device=self.device
        )
        out_num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        out_num_rejected = torch.empty(num_reqs, dtype=torch.int32, device=self.device)
        acc_nodes = self.step_acc_nodes[:num_reqs]
        total_logits, vocab_size = processed_logits.shape
        BLOCK_K = triton.next_power_of_2(self.top_k)
        BLOCK_V = (
            2048
            if vocab_size >= 2048
            else max(16, triton.next_power_of_2(vocab_size))
        )
        max_splits = max(1, triton.cdiv(vocab_size, BLOCK_V))

        # 1) Zero-copy fused split-V online LSE directly on bf16/fp32 logits across all SMs
        if total_logits <= 8:
            target_lse_splits = 32
        elif total_logits <= 32:
            target_lse_splits = 8
        elif total_logits <= 128:
            target_lse_splits = 4
        else:
            target_lse_splits = 2
        lse_splits = min(target_lse_splits, max_splits)
        if total_logits > self._lse_buf_cap:
            cap = max(total_logits, 256)
            self._row_lse = torch.empty(cap, dtype=torch.float32, device=self.device)
            self._lse_part_max = torch.empty(
                (cap, 32), dtype=torch.float32, device=self.device
            )
            self._lse_part_sum = torch.empty(
                (cap, 32), dtype=torch.float32, device=self.device
            )
            self._lse_buf_cap = cap
        row_lse = self._row_lse[:total_logits]
        # Use flat 1D views sized (total_logits * lse_splits) with zero allocation
        part_max = self._lse_part_max.view(-1)[: total_logits * lse_splits]
        part_sum = self._lse_part_sum.view(-1)[: total_logits * lse_splits]

        lse_split_size = triton.cdiv(vocab_size, lse_splits)
        lse_split_size = triton.cdiv(lse_split_size, BLOCK_V) * BLOCK_V
        _fused_row_lse_split_kernel[(total_logits, lse_splits)](
            processed_logits,
            processed_logits.stride(0),
            vocab_size,
            lse_split_size,
            part_max,
            part_sum,
            NUM_SPLITS=lse_splits,
            BLOCK_V=BLOCK_V,
            num_warps=8,
        )
        _reduce_row_lse_kernel[(total_logits,)](
            part_max,
            part_sum,
            row_lse,
            NUM_SPLITS=lse_splits,
            BLOCK_S=max(16, triton.next_power_of_2(lse_splits)),
            num_warps=1,
        )

        # 2) Register-only sparse recursive residual tree walk (no full-vocab scan!)
        _tree_verify_stochastic_kernel[(num_reqs,)](
            processed_logits,
            row_lse,
            processed_logits.stride(0),
            vocab_size,
            idx_mapping,
            cu_num_logits,
            positions,
            seeds,
            temperature,
            self.tree_tokens,
            self.tree_parents,
            self.tree_cand_ids,
            self.tree_cand_probs,
            self.tree_child_rank,
            self.tree_is_stochastic,
            out_sampled,
            acc_nodes,
            out_num_sampled,
            out_num_rejected,
            self._stop_row,
            self._stop_step,
            self._stop_u,
            self._stop_mode,
            BUDGET=self.budget,
            B1=self.b1,
            TOP_K=self.top_k,
            BLOCK_K=BLOCK_K,
            BLOCK_V=BLOCK_V,
            num_warps=2,
        )

        # 3) All-SM parallel vocab bonus sample for leaf / p_rest / prefill stopping nodes
        bonus_splits = min(32, max_splits)
        bonus_split_size = triton.cdiv(vocab_size, bonus_splits)
        bonus_split_size = triton.cdiv(bonus_split_size, BLOCK_V) * BLOCK_V
        b_part_val = self._bonus_part_val.view(-1)[: num_reqs * bonus_splits]
        b_part_idx = self._bonus_part_idx.view(-1)[: num_reqs * bonus_splits]
        _parallel_vocab_bonus_split_kernel[(num_reqs, bonus_splits)](
            processed_logits,
            processed_logits.stride(0),
            vocab_size,
            bonus_split_size,
            idx_mapping,
            positions,
            seeds,
            self.tree_cand_ids,
            self._stop_row,
            self._stop_u,
            self._stop_mode,
            b_part_val,
            b_part_idx,
            B1=self.b1,
            TOP_K=self.top_k,
            BONUS_SPLITS=bonus_splits,
            BLOCK_V=BLOCK_V,
            num_warps=8,
        )
        _reduce_vocab_bonus_kernel[(num_reqs,)](
            self._stop_mode,
            self._stop_step,
            b_part_val,
            b_part_idx,
            out_sampled,
            B1=self.b1,
            BONUS_SPLITS=bonus_splits,
            BLOCK_S=max(16, triton.next_power_of_2(bonus_splits)),
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
