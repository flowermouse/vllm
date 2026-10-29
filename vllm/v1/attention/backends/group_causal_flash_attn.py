# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention with a group-causal mask inside the draft's query block.

The draft block of E7 (scratch tokens, AngelSpec Scratch-Tokens-Impl.md section 10) is laid out as
``S`` groups of ``g = 1 + r`` tokens (a slot and its ``r`` scratch tokens). Group ``i`` attends to the
whole context and to groups ``0..i`` (bidirectionally inside a group), never to later groups, so
lengthening or shortening the chain does not change the groups before it.

That mask needs no new kernel. Every request is split into ``S`` pseudo-requests, one per group: the
``g`` query tokens of group ``i`` read the request's own block-table row up to
``context_len + g * (i + 1)`` cache tokens, non-causally. The K/V of all ``S * g`` query tokens are
written to the cache before attention (their slots are consecutive, right after the context), so a
pseudo-request sees exactly the context, the groups before its own and its own group. Everything else
(paged cache, FA3/FA4 kernel, CUDA graph capture) is the stock FlashAttention path: only the metadata
handed to ``FlashAttentionImpl.forward`` is rebuilt, with static shapes and no host sync.

The attention layer carries ``group_size`` (``g``) and ``query_len`` (``S * g``, the draft query tokens
per request); the draft model sets them when it builds the layer.
"""

import dataclasses

import torch

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)


class GroupCausalFlashAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:  # profiling run
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        group_size: int = layer.group_size
        query_len: int = layer.query_len
        num_groups = query_len // group_size
        seq_lens = attn_metadata.seq_lens  # [R]: context + query_len cache tokens
        num_reqs = seq_lens.shape[0]
        device = seq_lens.device

        groups = torch.arange(1, num_groups + 1, device=device, dtype=seq_lens.dtype)
        context_lens = (seq_lens - query_len).clamp_(min=0)
        # a padded request (seq_len 0) keeps length 0 in every pseudo-request
        pseudo_seq_lens = (
            (context_lens.unsqueeze(1) + group_size * groups.unsqueeze(0))
            * (seq_lens > 0).unsqueeze(1)
        ).reshape(-1)
        pseudo_query_start_loc = (
            torch.arange(num_reqs * num_groups + 1, device=device, dtype=torch.int32)
            * group_size
        )
        pseudo_block_table = attn_metadata.block_table.repeat_interleave(
            num_groups, dim=0
        )
        pseudo_metadata = dataclasses.replace(
            attn_metadata,
            max_query_len=group_size,
            query_start_loc=pseudo_query_start_loc,
            seq_lens=pseudo_seq_lens.to(seq_lens.dtype),
            block_table=pseudo_block_table,
            scheduler_metadata=None,
            causal=False,
        )
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            pseudo_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class GroupCausalFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_impl_cls() -> type[GroupCausalFlashAttentionImpl]:
        return GroupCausalFlashAttentionImpl
