# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator of the AngelSpec CRL drafter (``models/qwen3_dflash_crl.py``).

The DFlash step runs as it does for DFlash 2 (block of ``1 + N`` query tokens, top-K candidates per row, walk kernel). Only the
scores differ: instead of a low-rank edge term, the conditional refinement layer scores each row's candidates for each of the
first ``crl_pred_topk`` candidates of the row before (row 1: the anchor token). Its ``N * crl_pred_topk`` queries per request attend
to the last draft layer's paged K/V cache (context + the block's own rows, non-causal), through that layer's FlashAttention
implementation with a rebuilt query layout -- no new kernel, static shapes, no host sync, so the whole step stays in the draft graph.

At concurrency 32 the stage's glue (index / cat / expand copies, RMSNorm decomposed into pow / mean / rsqrt / mul, the candidate gather of the LM head)
was ~450 of its ~700 us per engine step; ``crl/fused.py`` replaces it by two Triton kernels (``_crl_prep_kernel``, ``_crl_score_kernel``) and the CUDA
norm ops, ~40% off the stage.
"""

import atexit
import dataclasses
import os

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.crl.fused import (
    MODE_NULL,
    MODE_TEMPERATURE,
    _calib_store_kernel,
    _crl_confidence_kernel,
    _crl_prep_kernel,
    _crl_score_kernel,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

logger = init_logger(__name__)


class CRLSpeculator(DFlash2Speculator):
    _speculator_name = "CRL"

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self._crl_attn_metadata = None
        # a measurement switch: score the plain logits and skip the refinement layer (the cost of the stage = the per-round time difference)
        self._skip_stage = os.environ.get("VLLM_CRL_SKIP_STAGE", "0") == "1"
        # the stage's glue as two Triton kernels + the CUDA norm ops (crl/fused.py); VLLM_CRL_FUSED=0 runs the PyTorch reference path (same numbers
        # up to bf16 rounding of the candidate scores)
        self._fused = os.environ.get("VLLM_CRL_FUSED", "1") == "1"
        # Dynamic verification: where the per-row confidence comes from. "estimator" = upstream's online logistic on logit(max q) (the default
        # path of DFlash 2); "temperature" = max softmax(S / tau) with one tau per request class; "null" = (1 - sigmoid(delta)) * max softmax(S)
        # with the Null Candidate probe (``null_probe.npz`` next to the draft weights). Only the last two are computed here.
        spec = vllm_config.speculative_config
        self._conf_mode = spec.crl_confidence
        self._tau_greedy = float(spec.crl_confidence_tau)
        self._tau_sampled = float(spec.crl_confidence_tau_sampled or spec.crl_confidence_tau)
        self._null_w: torch.Tensor | None = None
        self._null_beta: torch.Tensor | None = None
        self._null_delta: torch.Tensor | None = None
        # Calibration dump (a measurement tool, host syncs on every step: never in a timed run): for every verified round the walked score rows
        # [N, K], the request's temperature and the verdict (accepted / admitted drafts), so that the one temperature of the 'temperature'
        # confidence can be fitted offline on the real sampling path (Dynamic-Verification-Eval, "T=1"). VLLM_CRL_CALIB_DUMP=<file.npz>
        self._calib_path = os.environ.get("VLLM_CRL_CALIB_DUMP") or None
        self._calib_rows: list = []
        self._calib_scores: torch.Tensor | None = None
        self._calib_valid: torch.Tensor | None = None
        logger.info(
            "%s: %d rows, top-%d candidates, %d predecessors per row; at temperature > 0 the drafts are %s",
            self._speculator_name,
            self.num_speculative_steps,
            self.top_k,
            int(self.draft_model_config.hf_config.dflash_config["crl_pred_topk"]),
            "sampled from the cached proposal (draft_sample_method=probabilistic)"
            if self.use_draft_logits
            else "the argmax walk with one-hot proposals (draft_sample_method=greedy)",
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # the target's (frozen) input embedding is shared with the draft by now: tabulate W_e RMSNorm_e(E_in) once
        model.model.crl.build_inference_buffers(model.model.embed_tokens.weight)
        if self._calib_path is not None:
            self._calib_scores = torch.zeros(self.max_num_reqs, self.num_speculative_steps, self.top_k, dtype=torch.float32, device=self.device)
            self._calib_valid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
            atexit.register(self._calib_save)
            logger.info("%s: calibration dump -> %s", self._speculator_name, self._calib_path)
        if self.enable_adaptive_verification and self._conf_mode != "estimator":
            # the confidence is computed in the draft graph (below), not by the online estimator
            self.use_acceptance_estimator = False
            if self._conf_mode == "null":
                self._load_null_probe()
        return model

    def _load_null_probe(self) -> None:
        if not self._fused:
            raise ValueError("crl_confidence='null' needs the fused refinement stage (VLLM_CRL_FUSED=1)")
        # (an .npz, not a .safetensors: the weight loader of the draft model reads every *.safetensors of the directory as model weights)
        path = os.path.join(self.draft_model_config.model, "null_probe.npz")
        if not os.path.exists(path):
            raise ValueError(f"crl_confidence='null' needs the Null Candidate probe {path} (AngelSpec: tools/sim_trajectory.py --null-probe-train)")
        import numpy as np

        with np.load(path) as f:
            w, beta = torch.from_numpy(f["w"]).float(), torch.from_numpy(f["beta"]).float()
        d = self.draft_model_config.hf_config.hidden_size
        if w.shape != (d,) or beta.shape != (self.num_speculative_steps,):
            raise ValueError(f"{path}: w {tuple(w.shape)} / beta {tuple(beta.shape)} do not match hidden {d} / {self.num_speculative_steps} rows")
        self._null_w = w.to(self.device)
        self._null_beta = beta.to(self.device)
        ka = int(self.draft_model_config.hf_config.dflash_config["crl_pred_topk"])
        self._null_delta = torch.zeros(self.max_num_reqs, self.num_speculative_steps, ka, dtype=torch.float32, device=self.device)
        logger.info("%s: Null Candidate probe loaded from %s", self._speculator_name, path)

    def _calib_save(self) -> None:
        if not self._calib_rows:
            return
        import numpy as np

        cols = list(zip(*self._calib_rows))
        np.savez(
            self._calib_path,
            scores=np.concatenate(cols[0]),
            temperature=np.concatenate(cols[1]),
            accepted=np.concatenate(cols[2]),
            admitted=np.concatenate(cols[3]),
        )

    def observe_verification(self, idx_mapping, num_sampled, num_rejected) -> None:
        super().observe_verification(idx_mapping, num_sampled, num_rejected)
        if self._calib_path is None or self._calib_scores is None:
            return
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        with gpu_sync_allowed():
            ok = self._calib_valid[idx_mapping] > 0
            self._calib_valid[idx_mapping] = 0
            keep = ok.cpu().numpy()
            if keep.any():
                accepted = (num_sampled.long() - 1).clamp(min=0)
                admitted = accepted + num_rejected.long()
                self._calib_rows.append(
                    (
                        self._calib_scores[idx_mapping].cpu().numpy()[keep],
                        self.temperature[idx_mapping].cpu().numpy()[keep],
                        accepted.cpu().numpy()[keep],
                        admitted.cpu().numpy()[keep],
                    )
                )
                if len(self._calib_rows) % 400 == 0:
                    self._calib_save()

    def _predict_confidence(self, candidate_ids, num_reqs, num_sample) -> None:
        if self._conf_mode == "estimator":
            super()._predict_confidence(candidate_ids, num_reqs, num_sample)
            return
        if self._skip_stage or self._crl_attn_metadata is None:
            return  # a dummy run without attention metadata: no scores to read
        steps = self.num_speculative_steps
        null = self._conf_mode == "null"
        _crl_confidence_kernel[(num_reqs * steps,)](
            self.candidate_sampler.scores,
            candidate_ids.contiguous(),
            self.draft_tokens,
            self._null_delta if null else self.candidate_sampler.scores,
            self._null_beta if null else self.candidate_sampler.scores,
            self.sample_idx_mapping,
            self.temperature,
            self.draft_token_confidence_probs,
            self.draft_token_confidence_probs.stride(0),
            self._tau_greedy,
            self._tau_sampled,
            num_rows=steps,
            top_k=self.top_k,
            ka=int(self.draft_model_config.hf_config.dflash_config["crl_pred_topk"]),
            MODE=MODE_NULL if null else MODE_TEMPERATURE,
            num_warps=1,
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
        if self._calib_scores is not None and self._crl_attn_metadata is not None:
            steps = self.num_speculative_steps
            _calib_store_kernel[(num_reqs * steps,)](
                self.candidate_sampler.scores,
                self.sample_idx_mapping,
                self._calib_scores,
                self._calib_valid,
                num_rows=steps,
                top_k=self.top_k,
                num_warps=1,
            )

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """``[R, N, K, K]`` score tables: ``table[r, k, i, b] = S_k(cand_{k-1}[i], cand_k[b])`` for ``i < crl_pred_topk``; the other
        predecessors (and nothing else) fall back to the row's own logits. Row 1's table (index 0) is scored given the anchor token.

        A request carries ``1 + (N - 1) * crl_pred_topk`` refinement queries: row 1's only predecessor is the anchor token, so it is
        scored once (its table rows are identical copies), every later row once per predecessor."""
        model = self.model.model
        crl = model.crl
        num_reqs, steps, k = candidate_ids.shape
        if self._skip_stage or self._crl_attn_metadata is None:  # a profiling / dummy run without attention metadata: nothing to read
            return unary_logits.view(num_reqs, steps, 1, k).float().expand(-1, -1, k, -1).contiguous()
        if self._fused:
            return self._score_fused(candidate_ids, unary_logits, hidden_states)
        ka = crl.pred_topk
        d = model.config.hidden_size
        device = candidate_ids.device
        h_pre = hidden_states[..., d:]  # [R, N, D]: the rows' last-layer outputs before the final norm
        anchor = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        # Row 1 (index 0) has exactly one predecessor, the anchor token (the old layout scored it ka times with the same input and kept
        # identical copies), so a request carries 1 + (N - 1) * ka refinement queries: row 0 once, every later row once per predecessor.
        pred = torch.cat(
            (anchor.view(num_reqs, 1), candidate_ids[:, :-1, :ka].reshape(num_reqs, -1)),
            dim=1,
        )  # [R, Q]
        h = torch.cat(
            (
                h_pre[:, :1],
                h_pre[:, 1:, None, :].expand(-1, -1, ka, -1).reshape(num_reqs, -1, d),
            ),
            dim=1,
        ).reshape(-1, d)
        sample_pos = self.sample_pos[: num_reqs * steps].view(num_reqs, steps)
        positions = torch.cat(
            (sample_pos[:, :1], sample_pos[:, 1:, None].expand(-1, -1, ka).reshape(num_reqs, -1)),
            dim=1,
        ).reshape(-1)

        attn_mod = model.layers[-1].self_attn
        layer = attn_mod.attn
        metadata = self._crl_attn_metadata[layer.layer_name]
        queries_per_req = 1 + (steps - 1) * ka
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

        down, r = crl(h, pred.reshape(-1), rope, attend)  # [R * Q, D] each
        # final norm of r'' = r' + down: the add is fused into the norm (the base-class call, not the pre-norm variant)
        x, _ = RMSNorm.forward(model.norm, down, r)
        x = x.view(num_reqs, queries_per_req, d)
        lm_weight = self.model.lm_head.weight
        w = lm_weight[candidate_ids]  # [R, N, K, D]
        scores0 = torch.einsum("rd,rkd->rk", x[:, 0], w[:, 0]).float()
        scores = torch.einsum(
            "rnad,rnkd->rnak", x[:, 1:].reshape(num_reqs, steps - 1, ka, d), w[:, 1:]
        ).float()
        table = unary_logits.view(num_reqs, steps, 1, k).float().expand(-1, -1, k, -1).clone()
        table[:, 0] = scores0[:, None, :]
        table[:, 1:, :ka, :] = scores
        return table

    def _score_fused(self, candidate_ids, unary_logits, hidden_states):
        model = self.model.model
        crl = model.crl
        num_reqs, steps, k = candidate_ids.shape
        d = model.config.hidden_size
        device = candidate_ids.device
        ka = crl.pred_topk
        nq = 1 + (steps - 1) * ka
        total = num_reqs * nq
        hidden = hidden_states.reshape(num_reqs * steps, -1)
        # (a strided view of the input ids: the kernel indexes it as a dense [R] array)
        anchor = self.input_buffers.input_ids[: num_reqs * self.num_query_per_req : self.num_query_per_req].contiguous()
        candidate_ids = candidate_ids.contiguous()
        x1 = torch.empty(total, d, dtype=hidden.dtype, device=device)
        r = torch.empty_like(x1)
        positions = torch.empty(total, dtype=self.sample_pos.dtype, device=device)
        _crl_prep_kernel[(total,)](
            hidden,
            hidden.stride(0),
            anchor,
            candidate_ids,
            self.sample_pos,
            crl.emb_table,
            crl.norm1.weight,
            x1,
            r,
            positions,
            crl.norm1.variance_epsilon,
            num_rows=steps,
            top_k=k,
            ka=ka,
            D=d,
            BLOCK=triton.next_power_of_2(d),
        )
        q = crl.q_proj(x1)
        qh = q.view(-1, crl.head_dim)
        qn = torch.empty_like(qh)
        ops.rms_norm(qn, qh, crl.q_norm.weight, crl.q_norm.variance_epsilon)
        q = qn.view(total, -1)
        cache = model._rope_cos_sin_cache
        if cache.dtype != q.dtype:
            cache = cache.to(dtype=q.dtype)
        ops.rotary_embedding(positions, q, None, model._rope_head_size, cache, model._rope_is_neox)

        layer = model.layers[-1].self_attn.attn
        metadata = self._crl_attn_metadata[layer.layer_name]
        refinement_metadata = dataclasses.replace(
            metadata,
            num_actual_tokens=total,
            max_query_len=nq,
            query_start_loc=torch.arange(num_reqs + 1, device=device, dtype=metadata.query_start_loc.dtype) * nq,
            scheduler_metadata=None,
            causal=False,
        )
        heads = q.view(total, -1, crl.head_dim)
        o = torch.empty_like(heads)
        layer.impl.forward(layer, heads, heads, heads, layer.kv_cache, refinement_metadata, o)
        x2 = crl.o_proj(o.view(total, -1))
        ops.fused_add_rms_norm(x2, r, crl.norm2.weight, crl.norm2.variance_epsilon)
        down = crl.down_proj(crl.act(torch.nn.functional.linear(x2, crl.gate_up_weight)))
        ops.fused_add_rms_norm(down, r, model.norm.weight, model.norm.variance_epsilon)  # r'' = r' + down, normed
        table = unary_logits.view(num_reqs, steps, 1, k).float().expand(-1, -1, k, -1).clone()
        null = self._conf_mode == "null" and self.enable_adaptive_verification
        _crl_score_kernel[(total,)](
            down,
            candidate_ids,
            self.model.lm_head.weight,
            table,
            self._null_w if null else table,
            self._null_delta if null else table,
            num_rows=steps,
            top_k=k,
            ka=ka,
            D=d,
            BLOCK_D=min(1024, triton.next_power_of_2(d)),
            NULL=null,
        )
        return table
