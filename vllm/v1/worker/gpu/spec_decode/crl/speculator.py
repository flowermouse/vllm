# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator of the AngelSpec CRL drafter (``models/qwen3_dflash_crl.py``).

The DFlash step runs as it does for DFlash 2 (block of ``1 + N`` query tokens, top-K candidates per row, walk kernel). Only the
scores differ: instead of a low-rank edge term, the conditional refinement layer scores each row's candidates for each of the
first ``crl_pred_topk`` candidates of the row before (row 1: the anchor token). Its ``N * crl_pred_topk`` queries per request attend
to the last draft layer's paged K/V cache (context + the block's own rows, non-causal), through that layer's FlashAttention
implementation with a rebuilt query layout -- no new kernel, static shapes, no host sync, so the whole step stays in the draft graph.
"""

import dataclasses

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

logger = init_logger(__name__)


class CRLSpeculator(DFlash2Speculator):
    _speculator_name = "CRL"

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self._crl_attn_metadata = None
        logger.info(
            "%s: %d rows, top-%d candidates, %d predecessors per row",
            self._speculator_name,
            self.num_speculative_steps,
            self.top_k,
            int(self.draft_model_config.hf_config.dflash_config["crl_pred_topk"]),
        )

    def _generate_draft(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=None,
    ) -> None:
        # the refinement queries are scored against the draft layer's cache with this step's metadata
        self._crl_attn_metadata = attn_metadata
        kwargs = {}
        if cudagraph_runtime_mode is not None:
            kwargs["cudagraph_runtime_mode"] = cudagraph_runtime_mode
        super()._generate_draft(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            **kwargs,
        )

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """``[R, N, K, K]`` score tables: ``table[r, k, i, b] = S_k(cand_{k-1}[i], cand_k[b])`` for ``i < crl_pred_topk``; the other
        predecessors (and nothing else) fall back to the row's own logits. Row 1's table (index 0) is scored given the anchor token."""
        model = self.model.model
        crl = model.crl
        num_reqs, steps, k = candidate_ids.shape
        ka = crl.pred_topk
        d = model.config.hidden_size
        device = candidate_ids.device
        h_pre = hidden_states[..., d:].reshape(num_reqs * steps, 1, d)

        anchor = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        pred = torch.cat(
            (anchor[:, None, None].expand(-1, 1, ka), candidate_ids[:, :-1, :ka]), dim=1
        )  # [R, N, ka]
        emb = model.embed_tokens(pred.reshape(-1)).view(num_reqs * steps, ka, d)
        h = h_pre.expand(-1, ka, -1).reshape(-1, d)
        emb = emb.reshape(-1, d)
        positions = (
            self.sample_pos[: num_reqs * steps]
            .view(num_reqs * steps, 1)
            .expand(-1, ka)
            .reshape(-1)
        )

        attn_mod = model.layers[-1].self_attn
        layer = attn_mod.attn
        metadata = self._crl_attn_metadata[layer.layer_name]
        queries_per_req = steps * ka
        refinement_metadata = dataclasses.replace(
            metadata,
            num_actual_tokens=num_reqs * queries_per_req,
            max_query_len=queries_per_req,
            query_start_loc=torch.arange(
                num_reqs + 1, device=device, dtype=metadata.query_start_loc.dtype
            )
            * queries_per_req,
            scheduler_metadata=None,
            causal=False,
        )

        def rope(q: torch.Tensor) -> None:
            cache = model._rope_cos_sin_cache
            if cache.dtype != q.dtype:
                cache = cache.to(dtype=q.dtype)
            ops.rotary_embedding(
                positions, q, None, model._rope_head_size, cache, model._rope_is_neox
            )

        def attend(q: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(q)
            layer.impl.forward(
                layer, q, q, q, layer.kv_cache, refinement_metadata, out
            )
            return out

        r = crl(h, emb, rope, attend)  # [R * N * ka, D]
        x = RMSNormOnly.apply(model.norm, r).view(num_reqs, steps, ka, d)
        lm_weight = self.model.lm_head.weight
        w = lm_weight[candidate_ids]  # [R, N, K, D]
        scores = torch.einsum("rnad,rnkd->rnak", x, w).float()
        table = unary_logits.view(num_reqs, steps, 1, k).float().expand(-1, -1, k, -1).clone()
        table[:, :, :ka, :] = scores
        return table


class RMSNormOnly:
    """The final norm applied to ``r''`` alone (the model's ``norm`` is the one that also returns the pre-norm sum)."""

    @staticmethod
    def apply(norm: torch.nn.Module, r: torch.Tensor) -> torch.Tensor:
        x = r.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return x.to(r.dtype) * norm.weight
