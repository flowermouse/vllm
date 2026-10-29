# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GroupCausalFlashAttentionImpl against a dense reference (GPU).

A request has S groups of g query tokens (a slot and its scratch tokens) after its context. Group i must
see the context, the groups 0..i and nothing after them; the backend gets there with per-group
pseudo-requests on the stock FlashAttention path.
"""

import math
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.group_causal_flash_attn import (
    GroupCausalFlashAttentionImpl,
)

try:  # collected by pytest where it exists; also runs as a plain script (the evaluation env has no pytest)
    import pytest

    pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    _parametrize = pytest.mark.parametrize
except ImportError:

    def _parametrize(name, values):
        return lambda fn: fn


H, KV, D, BLOCK = 32, 8, 128, 16


def _reference(q, k_all, v_all, ctx, groups, g):
    """q [S*g, H, D]; k_all / v_all [ctx + S*g, KV, D]: dense attention with the group-causal mask."""
    s_len = groups * g
    k = k_all.repeat_interleave(H // KV, dim=1).float()
    v = v_all.repeat_interleave(H // KV, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / math.sqrt(D)
    qi = torch.arange(s_len, device=q.device) // g  # group of every query token
    ki = torch.arange(ctx + s_len, device=q.device)
    group_of_key = torch.where(ki < ctx, torch.full_like(ki, -1), (ki - ctx) // g)
    mask = group_of_key[None, :] <= qi[:, None]  # context (-1) always visible
    scores = scores.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


@_parametrize("group_size", [1, 2, 4])
def test_group_causal_matches_dense_reference(group_size):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    groups = 15
    query_len = groups * group_size
    ctx_lens = [37, 100, 5]
    num_reqs = len(ctx_lens) + 1  # the last request is graph padding (seq_len 0)
    impl = GroupCausalFlashAttentionImpl(
        num_heads=H,
        head_size=D,
        scale=1.0 / math.sqrt(D),
        num_kv_heads=KV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones(1, device=dev),
        _k_scale=torch.ones(1, device=dev),
        _v_scale=torch.ones(1, device=dev),
        group_size=group_size,
        query_len=query_len,
    )
    max_blocks = (max(ctx_lens) + query_len + BLOCK - 1) // BLOCK
    num_blocks = 1 + num_reqs * max_blocks
    phys = torch.zeros(num_blocks, BLOCK, KV, 2 * D, device=dev, dtype=torch.bfloat16)
    kv_cache = phys.transpose(1, 2)  # (blocks, KV, BLOCK, 2D), as the backend expects
    block_table = torch.zeros(num_reqs, max_blocks, device=dev, dtype=torch.int32)
    for r in range(num_reqs):
        block_table[r] = 1 + r * max_blocks + torch.arange(max_blocks, device=dev)

    def slot(r, pos):
        return int(block_table[r, pos // BLOCK]) * BLOCK + pos % BLOCK

    ctx_k, ctx_v = [], []
    for r, c in enumerate(ctx_lens):
        k = torch.randn(c, KV, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(c, KV, D, device=dev, dtype=torch.bfloat16)
        for pos in range(c):
            blk, off = divmod(slot(r, pos), BLOCK)
            phys[blk, off, :, :D] = k[pos]
            phys[blk, off, :, D:] = v[pos]
        ctx_k.append(k)
        ctx_v.append(v)

    q = torch.randn(num_reqs * query_len, H, D, device=dev, dtype=torch.bfloat16)
    key = torch.randn(num_reqs * query_len, KV, D, device=dev, dtype=torch.bfloat16)
    value = torch.randn(num_reqs * query_len, KV, D, device=dev, dtype=torch.bfloat16)
    slots = torch.full((num_reqs * query_len,), -1, device=dev, dtype=torch.int64)
    for r, c in enumerate(ctx_lens):
        for off in range(query_len):  # one cache slot per query token, after the context
            slots[r * query_len + off] = slot(r, c + off)
    impl.do_kv_cache_update(layer, key, value, kv_cache, slots)

    seq_lens = torch.tensor(
        [c + query_len for c in ctx_lens] + [0], device=dev, dtype=torch.int32
    )
    metadata = FlashAttentionMetadata(
        num_actual_tokens=num_reqs * query_len,
        max_query_len=query_len,
        query_start_loc=torch.arange(num_reqs + 1, device=dev, dtype=torch.int32)
        * query_len,
        max_seq_len=max(ctx_lens) + query_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slots,
        causal=False,
    )
    out = torch.zeros(num_reqs * query_len, H, D, device=dev, dtype=torch.bfloat16)
    impl.forward(layer, q, key, value, kv_cache, metadata, out)
    torch.cuda.synchronize()

    for r, c in enumerate(ctx_lens):
        sl = slice(r * query_len, (r + 1) * query_len)
        k_all = torch.cat([ctx_k[r], key[sl]])
        v_all = torch.cat([ctx_v[r], value[sl]])
        ref = _reference(q[sl], k_all, v_all, c, groups, group_size)
        err = (out[sl].float() - ref).abs().max().item()
        assert err < 3e-2, f"request {r} (ctx {c}): max abs error {err}"
    assert torch.isfinite(out[: len(ctx_lens) * query_len]).all()


if __name__ == "__main__":
    for g in (1, 2, 4):
        test_group_causal_matches_dense_reference(g)
        print(f"group_size={g}: OK")
