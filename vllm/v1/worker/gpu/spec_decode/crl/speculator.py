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

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.crl import elastic_plan
from vllm.v1.worker.gpu.spec_decode.crl.elastic_full import ElasticFullGraphs
from vllm.v1.worker.gpu.spec_decode.crl.fused import (
    MODE_NULL,
    MODE_TEMPERATURE,
    _calib_store_kernel,
    _crl_confidence_kernel,
    _crl_prep_kernel,
    _crl_prep_varlen_kernel,
    _crl_score_kernel,
    _crl_score_varlen_kernel,
    _pava4_kernel,
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
        self._tau_sampled = float(
            spec.crl_confidence_tau_sampled or spec.crl_confidence_tau
        )
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
        self.tree_state = None
        if self.tree_budget > 0:
            from vllm.v1.worker.gpu.spec_decode.crl.tree import (
                CRLTreeState,
                get_crl_tree_temp,
            )

            self.tree_state = CRLTreeState(
                max_num_reqs=self.max_num_reqs,
                budget=self.tree_budget,
                draft_rows=self.num_speculative_steps,
                top_k=self.top_k,
                device=device,
                max_branch=4,
                temp=get_crl_tree_temp(),
            )
            logger.info(
                "%s: Zero-Cost Conditional Tree enabled (budget=%d, draft_rows=%d, temp=%.2f)",
                self._speculator_name,
                self.tree_budget,
                self.num_speculative_steps,
                self.tree_state.temp,
            )
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
            self._calib_scores = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.top_k,
                dtype=torch.float32,
                device=self.device,
            )
            self._calib_valid = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            atexit.register(self._calib_save)
            logger.info(
                "%s: calibration dump -> %s", self._speculator_name, self._calib_path
            )
        if self.enable_adaptive_verification and self._conf_mode != "estimator":
            # the confidence is computed in the draft graph (below), not by the online estimator
            self.use_acceptance_estimator = False
            if self._conf_mode == "null":
                self._load_null_probe()
        return model

    def _load_null_probe(self) -> None:
        if not self._fused:
            raise ValueError(
                "crl_confidence='null' needs the fused refinement stage (VLLM_CRL_FUSED=1)"
            )
        # (an .npz, not a .safetensors: the weight loader of the draft model reads every *.safetensors of the directory as model weights)
        path = os.path.join(self.draft_model_config.model, "null_probe.npz")
        if not os.path.exists(path):
            raise ValueError(
                f"crl_confidence='null' needs the Null Candidate probe {path} (AngelSpec: tools/sim_trajectory.py --null-probe-train)"
            )
        import numpy as np

        with np.load(path) as f:
            w, beta = (
                torch.from_numpy(f["w"]).float(),
                torch.from_numpy(f["beta"]).float(),
            )
        d = self.draft_model_config.hf_config.hidden_size
        if (
            w.shape != (d,)
            or beta.ndim != 1
            or beta.shape[0] < self.num_speculative_steps
        ):
            raise ValueError(
                f"{path}: w {tuple(w.shape)} / beta {tuple(beta.shape)} do not match hidden {d} / {self.num_speculative_steps} rows"
            )
        if beta.shape[0] > self.num_speculative_steps:
            beta = beta[: self.num_speculative_steps].contiguous()
        self._null_w = w.to(self.device)
        self._null_beta = beta.to(self.device)
        ka = int(self.draft_model_config.hf_config.dflash_config["crl_pred_topk"])
        self._null_delta = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            ka,
            dtype=torch.float32,
            device=self.device,
        )
        logger.info(
            "%s: Null Candidate probe loaded from %s", self._speculator_name, path
        )

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
        # The next gate reads this. Depth 0 (nothing drafted yet) is a no-op.
        if elastic_plan.lambda_star() is not None and idx_mapping.numel():
            self._gpu_observe(idx_mapping, num_sampled, num_rejected)
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
        if self._varlen_d is not None:
            self._generate_draft_varlen(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            return
        if self.tree_state is not None:
            last_hidden_states = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode or CUDAGraphMode.NONE,
            )
            num_sample = num_reqs * self.num_speculative_steps
            hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
                num_reqs, self.num_speculative_steps, -1
            )
            candidate_ids, unary_logits = self.model.compute_candidates(
                hidden_states.flatten(0, 1)
            )
            candidate_ids = candidate_ids.view(
                num_reqs, self.num_speculative_steps, self.top_k
            )
            unary_logits = unary_logits.view_as(candidate_ids)
            scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
            self.tree_state.build_trees(
                candidate_ids,
                scores,
                num_reqs,
                self.sample_idx_mapping,
                self.draft_tokens,
                pad_token=self.parallel_drafting_token_id,
                null_delta=(
                    self._null_delta
                    if (self._null_delta is not None and self._null_beta is not None)
                    else None
                ),
                null_beta=(
                    self._null_beta
                    if (self._null_delta is not None and self._null_beta is not None)
                    else None
                ),
                out_node_weights=(
                    self.draft_token_confidence_probs
                    if self.enable_adaptive_verification
                    else None
                ),
                temperature=self.temperature,
                seeds=self.seeds,
                sample_positions=self.sample_pos,
            )
            return
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

    def elastic_query_lengths(self, num_reqs: int, **kwargs) -> np.ndarray | None:
        """Query rows per request. A fixed env assignment is the 0b measurement; the gate is the serving path."""
        mode = os.environ.get("VLLM_ELASTIC_G", "")
        if mode:
            if mode == "all2":
                g = np.full(num_reqs, 2, dtype=np.int32)
            elif mode == "half12":
                g = np.empty(num_reqs, dtype=np.int32)
                half = num_reqs // 2
                g[:half] = 1
                g[half:] = 2
            elif mode == "all0":
                g = np.zeros(num_reqs, dtype=np.int32)
            elif mode == "alt715":
                # Odd/even depths 7, 15, 7, 15, ... (q = 8, 16). Not a contiguous split.
                g = np.where((np.arange(num_reqs) & 1) == 0, 2, 4).astype(np.int32)
            elif mode == "alt37":
                # Average g = 1.5: depths 3, 7, 3, 7, ... (q = 4, 8).
                g = np.where((np.arange(num_reqs) & 1) == 0, 1, 2).astype(np.int32)
            elif mode == "cycle3715":
                # Depths 3, 7, 11, 15 cycling (q = 4, 8, 12, 16).
                g = np.resize(np.asarray([1, 2, 3, 4], dtype=np.int32), num_reqs).copy()
            else:
                raise ValueError(
                    f"VLLM_ELASTIC_G={mode!r} is not all2, half12, all0, alt715, alt37, or cycle3715"
                )
            if not getattr(self, "_elastic_logged", False):
                mean_g = float(g.mean()) if num_reqs else 0.0
                logger.info(
                    "CRL elastic draft %s over %d requests, mean g=%.3f",
                    mode,
                    num_reqs,
                    mean_g,
                )
                self._elastic_logged = True
            return (4 * g).astype(np.int32)
        if os.environ.get("VLLM_ELASTIC", "") != "1":
            return None
        ef = getattr(self, "_elastic_full", None)
        if (
            ef is not None
            and ef._has_gate
            and elastic_plan.verify_cost_ready()
            and num_reqs > 0
            and os.environ.get("VLLM_ELASTIC_SYNC", "") != "1"
        ):
            self._elastic_use_async = True
            return np.zeros(0, dtype=np.int32)
        self._elastic_use_async = False
        return self._elastic_gate_lengths(num_reqs, **kwargs)

    def elastic_draft_counts(self, idx_mapping) -> np.ndarray | None:
        """Per-request draft counts from the last elastic propose. Zeros stay zeros.

        Pass the batch's ``idx_mapping_np``. A GPU tensor would sync the step.
        """
        if torch.is_tensor(idx_mapping):
            idx_mapping = idx_mapping.detach().cpu().numpy()
        ef = getattr(self, "_elastic_full", None)
        if ef is not None and ef._async_active:
            slots = np.asarray(idx_mapping, dtype=np.int64)
            if (
                self.enable_adaptive_verification
                and elastic_plan.verify_mode() != "off"
            ):
                return np.where(
                    ef._host_decode[slots], self.num_speculative_steps, 0
                ).astype(np.int32)
            return ef._host_d[slots]
        return elastic_plan.drafted_counts(np.asarray(idx_mapping))

    def _load_anchor_gate(self) -> dict[str, torch.Tensor] | None:
        gate = getattr(self, "_anchor_gate", None)
        if gate is not None or getattr(self, "_anchor_gate_missing", False):
            return gate
        path = os.environ.get("VLLM_ANCHOR_GATE", "")
        if not path:
            self._anchor_gate_missing = True
            logger.info(
                "VLLM_ELASTIC=1 but VLLM_ANCHOR_GATE is unset; keeping the fixed block"
            )
            return None
        loaded = np.load(path)
        keys = ["W_c", "W_e", "W_o", "b_o", "C"]
        if "W_s" in loaded:
            keys += ["W_s", "b_s", "W_su"]
        gate = {
            key: torch.tensor(loaded[key], device=self.device, dtype=torch.float32)
            for key in keys
        }
        self._anchor_gate = gate
        logger.info("loaded anchor gate %s", path)
        return gate

    def _elastic_p_next(self, input_batch, num_reqs: int, prefill: np.ndarray) -> int:
        if not prefill.any():
            return 0
        prefill_len = getattr(input_batch, "prefill_len_np", None)
        computed = getattr(input_batch, "num_computed_prefill_tokens_np", None)
        qsl = getattr(input_batch, "query_start_loc_np", None)
        chunk = (
            np.diff(qsl[: num_reqs + 1]).astype(np.int64)
            if qsl is not None
            else np.ones(num_reqs, np.int64)
        )
        if prefill_len is not None and computed is not None:
            remaining = np.maximum(prefill_len[:num_reqs] - computed[:num_reqs], 0)
        else:
            remaining = chunk
        return int(np.minimum(remaining, np.maximum(chunk, 1))[prefill].sum())

    def _capture_elastic_buckets(self) -> None:
        graphs = getattr(self, "_elastic_full", None)
        if graphs is None:
            graphs = ElasticFullGraphs(self)
            self._elastic_full = graphs
        graphs.capture_all()

    def _elastic_full_replay(
        self,
        q_lens,
        input_batch,
        num_reqs,
        num_target_tokens,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dummy_run,
    ) -> bool:
        graphs = getattr(self, "_elastic_full", None)
        if graphs is None or not graphs.graphs:
            self._elastic_full_misses = getattr(self, "_elastic_full_misses", 0) + 1
            logger.info(
                "ELASTIC_FULL_MISS q_tot=%s no-graph", int(np.asarray(q_lens).sum())
            )
            return False
        return graphs.replay(
            q_lens,
            input_batch,
            num_reqs,
            num_target_tokens,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dummy_run,
        )

    def _ensure_gpu_traj(self) -> None:
        if getattr(self, "_gpu_tau", None) is not None:
            return
        n = self.max_num_reqs
        device = self.device
        self._gpu_req = torch.full((n,), -1, dtype=torch.int64, device=device)
        self._gpu_tau = torch.zeros(n, device=device)
        self._gpu_ema = torch.zeros(n, device=device)
        self._gpu_hit = torch.zeros(n, device=device)
        self._gpu_q = torch.full((n,), 0.5, device=device)
        self._gpu_steps = torch.zeros(n, dtype=torch.int32, device=device)
        self._gpu_d = torch.zeros(n, dtype=torch.int32, device=device)
        self._gpu_cols = torch.arange(15, device=device)

    def _gpu_bind_state(self, slots: torch.Tensor, req_ids: np.ndarray) -> torch.Tensor:
        self._ensure_gpu_traj()
        r = int(slots.shape[0])
        cpu_buf = getattr(self, "_req_ids_cpu", None)
        if cpu_buf is None or cpu_buf.shape[0] < self.max_num_reqs:
            cpu_buf = torch.empty(self.max_num_reqs, dtype=torch.int64, pin_memory=True)
            self._req_ids_cpu = cpu_buf
        cpu_buf[:r].copy_(torch.from_numpy(req_ids))
        req = cpu_buf[:r].to(slots.device, non_blocking=True)
        return self._gpu_bind_state_dev(slots, req)

    def _gpu_bind_state_dev(
        self, slots: torch.Tensor, req: torch.Tensor, live_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        import triton

        from vllm.v1.worker.gpu.spec_decode.crl.fused import _traj_bind_state_kernel

        self._ensure_gpu_traj()
        r = int(slots.shape[0])
        out = torch.empty((r, 5), dtype=torch.float32, device=slots.device)
        block = 128
        grid = (triton.cdiv(r, block),)
        _traj_bind_state_kernel[grid](
            slots,
            req,
            self._gpu_req,
            self._gpu_tau,
            self._gpu_ema,
            self._gpu_hit,
            self._gpu_q,
            self._gpu_steps,
            self._gpu_d,
            out,
            live_t if live_t is not None else self._gpu_d,
            r,
            HAS_LIVE=1 if live_t is not None else 0,
            BLOCK=block,
        )
        return out

    def _gpu_bind(self, slots: torch.Tensor, req_ids: np.ndarray) -> None:
        self._ensure_gpu_traj()
        req = torch.as_tensor(req_ids, device=slots.device)
        prev = self._gpu_req[slots]
        changed = prev != req
        zero = torch.zeros(slots.shape[0], device=slots.device)
        half = torch.full((slots.shape[0],), 0.5, device=slots.device)
        zint = torch.zeros(slots.shape[0], dtype=torch.int32, device=slots.device)
        self._gpu_tau[slots] = torch.where(changed, zero, self._gpu_tau[slots])
        self._gpu_ema[slots] = torch.where(changed, zero, self._gpu_ema[slots])
        self._gpu_hit[slots] = torch.where(changed, zero, self._gpu_hit[slots])
        self._gpu_q[slots] = torch.where(changed, half, self._gpu_q[slots])
        self._gpu_steps[slots] = torch.where(changed, zint, self._gpu_steps[slots])
        self._gpu_d[slots] = torch.where(changed, zint, self._gpu_d[slots])
        self._gpu_req[slots] = req

    def _gpu_state(self, slots: torch.Tensor) -> torch.Tensor:
        self._ensure_gpu_traj()
        return torch.stack(
            (
                self._gpu_tau[slots] / 15.0,
                self._gpu_ema[slots] / 15.0,
                self._gpu_hit[slots],
                self._gpu_q[slots],
                (self._gpu_steps[slots] == 0).to(torch.float32),
            ),
            dim=-1,
        )

    def _gpu_observe(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        """Fold the verified draft into on-device trajectory state. No D2H."""
        import triton

        from vllm.v1.worker.gpu.spec_decode.crl.fused import _gpu_observe_kernel

        self._ensure_gpu_traj()
        n = int(idx_mapping.shape[0])
        if n == 0:
            return
        conf = self.draft_token_confidence_probs
        block = 128
        _gpu_observe_kernel[(triton.cdiv(n, block),)](
            idx_mapping,
            num_sampled,
            num_rejected,
            conf,
            conf.stride(0),
            self._gpu_tau,
            self._gpu_ema,
            self._gpu_hit,
            self._gpu_q,
            self._gpu_steps,
            self._gpu_d,
            n,
            BLOCK=block,
        )

    def _elastic_gate_lengths(self, num_reqs: int, **kwargs) -> np.ndarray | None:
        """g_i from the anchor gate, then the pure-decode opportunity cost."""
        # Graph capture runs before the verify-cost table exists. Entering here
        # would price verification at the fallback and consume the one-shot log.
        if not elastic_plan.verify_cost_ready():
            return None
        gate = self._load_anchor_gate()
        aux = kwargs.get("aux_hidden_states")
        if gate is None or aux is None or num_reqs == 0:
            return None
        input_batch = kwargs["input_batch"]
        num_rejected = kwargs["num_rejected"]
        num_sampled = kwargs["num_sampled"]
        last_sampled = kwargs["last_sampled"]
        ends = (
            (
                input_batch.query_start_loc[1 : num_reqs + 1]
                - num_rejected[:num_reqs]
                - 1
            )
            .to(torch.long)
            .clamp(min=0)
        )
        fused = torch.cat([hidden.index_select(0, ends) for hidden in aux], dim=-1)
        # combine_hidden_states is fc only. The gate was trained on
        # extract_context_feature, which is fc followed by hidden_norm.
        projected = self.model.combine_hidden_states(fused)
        context = self.model.model.hidden_norm(projected).reshape(num_reqs, -1)
        slots = input_batch.idx_mapping[:num_reqs]
        slot_ids = np.asarray(input_batch.idx_mapping_np[:num_reqs], dtype=np.int64)
        req_ids = np.fromiter(
            (
                hash(input_batch.req_ids[i]) & 0x7FFFFFFFFFFFFFFF
                for i in range(num_reqs)
            ),
            dtype=np.int64,
            count=num_reqs,
        )
        embed = self.model.embed_input_ids(last_sampled[slots].reshape(-1)).reshape(
            num_reqs, -1
        )
        ctx = context.float()
        emb = embed.float()
        if "W_s" in gate:
            state = self._gpu_bind_state(slots, req_ids)
            h_s = torch.nn.functional.silu(state @ gate["W_s"].T + gate["b_s"])
            hidden = torch.nn.functional.silu(
                ctx @ gate["W_c"].T + emb @ gate["W_e"].T + h_s @ gate["W_su"].T
            )
            score = torch.cat((hidden, h_s), dim=-1) @ gate["W_o"].T + gate["b_o"]
        else:
            self._gpu_bind(slots, req_ids)
            hidden = torch.nn.functional.silu(ctx @ gate["W_c"].T + emb @ gate["W_e"].T)
            score = hidden @ gate["W_o"].T + gate["b_o"]
        if score.shape != (num_reqs, 4):
            raise RuntimeError(
                f"anchor gate returned {tuple(score.shape)} for {num_reqs} requests"
            )
        # PAVA and the auction stay on device. One small copy brings q back
        # for the scheduler; nothing else in the step syncs.
        raw = (gate["C"] * torch.sigmoid(score)).contiguous()
        delta = torch.empty_like(raw)
        _pava4_kernel[(num_reqs,)](raw, delta)
        decode = num_sampled[:num_reqs] > 0
        computed = getattr(input_batch, "num_computed_prefill_tokens_np", None)
        prefill_len = getattr(input_batch, "prefill_len_np", None)
        if computed is not None and prefill_len is not None:
            prefill_np = (prefill_len[:num_reqs] - computed[:num_reqs]) > 0
        else:
            prefill_np = np.zeros(num_reqs, dtype=bool)
        p_next = self._elastic_p_next(input_batch, num_reqs, prefill_np)
        hit = self._gpu_hit[slots] > 0.5
        prev_d = self._gpu_d[slots]
        cold = self._gpu_steps[slots] == 0
        query_t, lam_t = elastic_plan.choose_total_torch(
            delta, decode, p_next, hit, prev_d, cold
        )
        depth_t = torch.where(query_t > 0, query_t - 1, torch.zeros_like(query_t))
        self._gpu_d[slots] = depth_t
        # Replay reads this buffer. Stage non-gate inputs before the D2H sync so
        # the GPU executes them while the host waits, and defer publish() until
        # after replay() is launched.
        self._elastic_q_gpu = query_t
        ef = getattr(self, "_elastic_full", None)
        if ef is not None and kwargs.get("last_hidden_states") is not None:
            ef.pre_stage(
                input_batch,
                num_reqs,
                kwargs["num_target_tokens"],
                kwargs["last_hidden_states"],
                aux,
                num_sampled,
                num_rejected,
                last_sampled,
                kwargs["next_prefill_tokens"],
                kwargs["temperature"],
                kwargs["seeds"],
            )
        packed = torch.empty(num_reqs + 1, device=query_t.device, dtype=torch.float32)
        packed[:num_reqs] = query_t.float()
        packed[num_reqs] = lam_t
        host = packed.detach().cpu().numpy()
        query = host[:num_reqs].astype(np.int32)
        lam = float(host[-1])
        self._pending_elastic_publish = (slot_ids, query, lam)
        if not getattr(self, "_elastic_logged", False):
            logger.info(
                "CRL elastic gate over %d requests, mean g=%.3f lambda=%.3f p_next=%d",
                num_reqs,
                float((query // 4).mean()) if num_reqs else 0.0,
                lam,
                p_next,
            )
            self._elastic_logged = True
        return query

    def _flush_elastic_publish(self) -> None:
        pending = getattr(self, "_pending_elastic_publish", None)
        if pending is not None:
            self._pending_elastic_publish = None
            slot_ids, query, lam = pending
            elastic_plan.publish(slot_ids, query, lam, self.max_num_reqs)

    def _generate_draft_varlen(
        self,
        num_reqs,
        num_tokens_padded,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=None,
    ) -> None:
        """Backbone over the packed rows; lm_head and CRL only on the drafted mask rows."""
        kwargs = {}
        if cudagraph_runtime_mode is not None:
            kwargs["cudagraph_runtime_mode"] = cudagraph_runtime_mode
        # The parent writes self.draft_tokens from a rectangular walk. Run the
        # backbone here and score only the rows this request actually drafted.
        d_lens = self._varlen_d
        assert d_lens is not None
        # One depth value means the private uniform graph may replay. Mixed
        # depths, including a zero, stay on the compiled varlen forward.
        uniform = int(d_lens.min()) > 0 and int(d_lens.min()) == int(d_lens.max())
        last_hidden = self._run_model_elastic(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            kwargs.get("cudagraph_runtime_mode", CUDAGraphMode.NONE),
            uniform=uniform,
        )
        steps = self.num_speculative_steps
        k = self.top_k
        device = last_hidden.device
        cand = torch.zeros(num_reqs, steps, k, dtype=torch.int64, device=device)
        unary = torch.full(
            (num_reqs, steps, k), float("-inf"), dtype=torch.float32, device=device
        )
        sample_idx = self.sample_indices[: num_reqs * steps].view(num_reqs, steps)
        d_gpu, _pack = self._varlen_pack(d_lens, device)
        cols = self._varlen_cols(device)
        valid = cols.unsqueeze(0) < d_gpu.unsqueeze(1)
        if int(d_lens.sum()) > 0:
            flat_idx = sample_idx.reshape(-1)[valid.reshape(-1)]
            ids, logits = self.model.compute_candidates(
                last_hidden.index_select(0, flat_idx)
            )
            cand.view(-1, k)[valid.reshape(-1)] = ids
            unary.view(-1, k)[valid.reshape(-1)] = logits.float()
        table = unary[:, :, None, :].expand(-1, -1, k, -1).clone()
        sample_pos = self.sample_pos[: num_reqs * steps].view(num_reqs, steps)
        anchors = self.input_buffers.input_ids[self._query_cu_gpu[:num_reqs]]
        self._score_packed(
            cand, unary, last_hidden, sample_idx, sample_pos, anchors, d_lens, table
        )
        limits = d_gpu.to(torch.int32)
        self.draft_tokens[:num_reqs].fill_(self.parallel_drafting_token_id)
        self.candidate_sampler.sample(
            cand,
            table,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
            self.draft_logits_num_shards,
            self.draft_logits_shard_idx,
            row_limits=limits,
            pad_token=self.parallel_drafting_token_id,
        )
        if self.enable_adaptive_verification:
            # Undrafted rows are -inf. Softmax of that is NaN, and a 0 left in
            # the tail makes the next step's cumprod die at the previous depth.
            # Keep a continuation of 1 and skip those rows in the kernel.
            conf = self.draft_token_confidence_probs[:num_reqs]
            conf.fill_(1)
            mapping = self.sample_idx_mapping[: num_reqs * steps].view(num_reqs, steps)
            mapping.masked_fill_(~valid, -1)
            self._predict_confidence(cand, num_reqs, num_reqs * steps)
            conf.masked_fill_(~torch.isfinite(conf), 1)

    def _varlen_cols(self, device: torch.device) -> torch.Tensor:
        cols = getattr(self, "_varlen_cols_buf", None)
        if cols is None or cols.device != device:
            cols = torch.arange(self.num_speculative_steps, device=device)
            self._varlen_cols_buf = cols
        return cols

    def _varlen_pack(self, d_lens: np.ndarray, device: torch.device):
        """GPU index tensors for one depth layout. Reused while the layout holds.

        Building these from numpy every step syncs the host inside the drafter.
        """
        d_np = np.ascontiguousarray(d_lens, dtype=np.int32)
        ka = int(self.model.model.crl.pred_topk)
        key = (d_np.tobytes(), ka, str(device))
        cached = getattr(self, "_varlen_pack_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        nq_np = np.where(d_np > 0, 1 + (d_np - 1) * ka, 0).astype(np.int32)
        row_cu_np = np.cumsum(d_np, dtype=np.int32) - d_np
        q_cu_np = np.zeros(d_np.shape[0] + 1, dtype=np.int32)
        np.cumsum(nq_np, out=q_cu_np[1:])
        req_of = np.repeat(np.arange(d_np.shape[0], dtype=np.int32), nq_np)
        q_local = (
            np.concatenate([np.arange(int(n), dtype=np.int32) for n in nq_np if n > 0])
            if int(nq_np.sum())
            else np.zeros(0, dtype=np.int32)
        )
        pack = {
            "total": int(nq_np.sum()),
            "max_nq": int(nq_np.max()) if nq_np.size else 0,
            "req": torch.as_tensor(req_of, device=device),
            "qlocal": torch.as_tensor(q_local, device=device),
            "row_cu": torch.as_tensor(row_cu_np, device=device),
            "q_cu": torch.as_tensor(q_cu_np, device=device),
        }
        d_gpu = torch.as_tensor(d_np, device=device)
        self._varlen_pack_cache = (key, d_gpu, pack)
        return d_gpu, pack

    def _score_packed(
        self, cand, unary, last_hidden, sample_idx, sample_pos, anchors, d_lens, table
    ) -> None:
        """One prep, one attention and one score over every drafted row.

        ``d_lens`` is per request and may be 0. Queries are packed; ``_null_delta``
        is written on the full ``[R, 15, ka]`` stride so runs cannot overlap.
        """
        del unary  # candidates were already gathered into ``cand``
        model = self.model.model
        crl = model.crl
        ka = int(crl.pred_topk)
        width = model.config.hidden_size
        k = self.top_k
        steps = self.num_speculative_steps
        device = cand.device
        num_reqs = cand.shape[0]
        _d_gpu, pack = self._varlen_pack(d_lens, device)
        total = pack["total"]
        if total == 0 or self._crl_attn_metadata is None:
            return
        cols = self._varlen_cols(device)
        valid = cols.unsqueeze(0) < _d_gpu.unsqueeze(1)
        # Same row-major order as ``row_cu``: request 0's drafted rows, then request 1's.
        hidden = last_hidden.index_select(0, sample_idx.reshape(-1)[valid.reshape(-1)])
        x1 = torch.empty(total, width, dtype=hidden.dtype, device=device)
        residual = torch.empty_like(x1)
        positions = torch.empty(total, dtype=sample_pos.dtype, device=device)
        block = triton.next_power_of_2(width)
        _crl_prep_varlen_kernel[(total,)](
            hidden,
            hidden.stride(0),
            anchors.contiguous(),
            cand.contiguous(),
            sample_pos.contiguous(),
            crl.emb_table,
            crl.norm1.weight,
            x1,
            residual,
            positions,
            pack["req"],
            pack["qlocal"],
            pack["row_cu"],
            crl.norm1.variance_epsilon,
            STEPS=steps,
            top_k=k,
            ka=ka,
            D=width,
            BLOCK=block,
        )
        q = crl.q_proj(x1)
        qh = q.view(-1, crl.head_dim)
        qn = torch.empty_like(qh)
        ops.rms_norm(qn, qh, crl.q_norm.weight, crl.q_norm.variance_epsilon)
        q = qn.view(total, -1)
        cache = model._rope_cos_sin_cache
        if cache.dtype != q.dtype:
            cache = cache.to(dtype=q.dtype)
        ops.rotary_embedding(
            positions, q, None, model._rope_head_size, cache, model._rope_is_neox
        )
        layer = model.layers[-1].self_attn.attn
        metadata = self._crl_attn_metadata[layer.layer_name]
        refinement = dataclasses.replace(
            metadata,
            num_actual_tokens=total,
            max_query_len=pack["max_nq"],
            query_start_loc=pack["q_cu"],
            seq_lens=metadata.seq_lens[:num_reqs].contiguous(),
            block_table=metadata.block_table[:num_reqs].contiguous(),
            scheduler_metadata=None,
            causal=False,
        )
        heads = q.view(total, -1, crl.head_dim)
        out = torch.empty_like(heads)
        layer.impl.forward(layer, heads, heads, heads, layer.kv_cache, refinement, out)
        x2 = crl.o_proj(out.view(total, -1))
        ops.fused_add_rms_norm(
            x2, residual, crl.norm2.weight, crl.norm2.variance_epsilon
        )
        down = crl.down_proj(
            crl.act(torch.nn.functional.linear(x2, crl.gate_up_weight))
        )
        ops.fused_add_rms_norm(
            down, residual, model.norm.weight, model.norm.variance_epsilon
        )
        null = self._conf_mode == "null" and self.enable_adaptive_verification
        if null:
            assert self._null_delta is not None
            self._null_delta[:num_reqs].zero_()
            delta = self._null_delta
        else:
            delta = table
        _crl_score_varlen_kernel[(total,)](
            down,
            cand.contiguous(),
            self.model.lm_head.weight,
            table,
            self._null_w if null else table,
            delta,
            pack["req"],
            pack["qlocal"],
            STEPS=steps,
            top_k=k,
            ka=ka,
            D=width,
            BLOCK_D=min(1024, triton.next_power_of_2(width)),
            NULL=null,
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
        if (
            self._skip_stage or self._crl_attn_metadata is None
        ):  # a profiling / dummy run without attention metadata: nothing to read
            return (
                unary_logits.view(num_reqs, steps, 1, k)
                .float()
                .expand(-1, -1, k, -1)
                .contiguous()
            )
        if self._fused:
            return self._score_fused(candidate_ids, unary_logits, hidden_states)
        ka = crl.pred_topk
        d = model.config.hidden_size
        device = candidate_ids.device
        h_pre = hidden_states[
            ..., d:
        ]  # [R, N, D]: the rows' last-layer outputs before the final norm
        anchor = self.input_buffers.input_ids[
            : num_reqs * self.num_query_per_req : self.num_query_per_req
        ]
        # Row 1 (index 0) has exactly one predecessor, the anchor token (the old layout scored it ka times with the same input and kept
        # identical copies), so a request carries 1 + (N - 1) * ka refinement queries: row 0 once, every later row once per predecessor.
        pred = torch.cat(
            (
                anchor.view(num_reqs, 1),
                candidate_ids[:, :-1, :ka].reshape(num_reqs, -1),
            ),
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
            (
                sample_pos[:, :1],
                sample_pos[:, 1:, None].expand(-1, -1, ka).reshape(num_reqs, -1),
            ),
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
            layer.impl.forward(layer, q, q, q, layer.kv_cache, refinement_metadata, out)
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
        table = (
            unary_logits.view(num_reqs, steps, 1, k)
            .float()
            .expand(-1, -1, k, -1)
            .clone()
        )
        table[:, 0] = scores0[:, None, :]
        table[:, 1:, :ka, :] = scores
        return table

    def _score_fused(
        self,
        candidate_ids,
        unary_logits,
        hidden_states,
        anchor=None,
        sample_pos=None,
        req_range: tuple[int, int] | None = None,
    ):
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
        if anchor is None:
            anchor = self.input_buffers.input_ids[
                : num_reqs * self.num_query_per_req : self.num_query_per_req
            ]
        anchor = anchor.contiguous()
        if sample_pos is None:
            sample_pos = self.sample_pos
        sample_pos = sample_pos.contiguous()
        candidate_ids = candidate_ids.contiguous()
        x1 = torch.empty(total, d, dtype=hidden.dtype, device=device)
        r = torch.empty_like(x1)
        positions = torch.empty(total, dtype=self.sample_pos.dtype, device=device)
        _crl_prep_kernel[(total,)](
            hidden,
            hidden.stride(0),
            anchor,
            candidate_ids,
            sample_pos,
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
        ops.rotary_embedding(
            positions, q, None, model._rope_head_size, cache, model._rope_is_neox
        )

        layer = model.layers[-1].self_attn.attn
        metadata = self._crl_attn_metadata[layer.layer_name]
        seq_lens = metadata.seq_lens
        block_table = metadata.block_table
        if req_range is not None:
            start, end = req_range
            seq_lens = metadata.seq_lens[start:end].contiguous()
            block_table = metadata.block_table[start:end].contiguous()
        refinement_metadata = dataclasses.replace(
            metadata,
            num_actual_tokens=total,
            max_query_len=nq,
            query_start_loc=torch.arange(
                num_reqs + 1, device=device, dtype=metadata.query_start_loc.dtype
            )
            * nq,
            seq_lens=seq_lens,
            block_table=block_table,
            scheduler_metadata=None,
            causal=False,
        )
        heads = q.view(total, -1, crl.head_dim)
        o = torch.empty_like(heads)
        layer.impl.forward(
            layer, heads, heads, heads, layer.kv_cache, refinement_metadata, o
        )
        x2 = crl.o_proj(o.view(total, -1))
        ops.fused_add_rms_norm(x2, r, crl.norm2.weight, crl.norm2.variance_epsilon)
        down = crl.down_proj(
            crl.act(torch.nn.functional.linear(x2, crl.gate_up_weight))
        )
        ops.fused_add_rms_norm(
            down, r, model.norm.weight, model.norm.variance_epsilon
        )  # r'' = r' + down, normed
        table = (
            unary_logits.view(num_reqs, steps, 1, k)
            .float()
            .expand(-1, -1, k, -1)
            .clone()
        )
        null = self._conf_mode == "null" and self.enable_adaptive_verification
        # The kernel addresses delta as a dense [R_local, num_rows, ka] buffer
        # that starts at request 0. A run is a slice of the batch and num_rows
        # is that run's depth, not 15, so a direct store into self._null_delta
        # (stride 15, origin request 0) overlaps other runs. The confidence
        # kernel then reads with stride 15.
        if null:
            delta = torch.zeros(num_reqs, steps, ka, dtype=torch.float32, device=device)
        else:
            delta = table
        _crl_score_kernel[(total,)](
            down,
            candidate_ids,
            self.model.lm_head.weight,
            table,
            self._null_w if null else table,
            delta,
            num_rows=steps,
            top_k=k,
            ka=ka,
            D=d,
            BLOCK_D=min(1024, triton.next_power_of_2(d)),
            NULL=null,
        )
        if null:
            assert self._null_delta is not None
            if req_range is None:
                self._null_delta[:num_reqs, :steps].copy_(delta)
            else:
                start, end = req_range
                self._null_delta[start:end, :steps].copy_(delta)
        return table


def _equal_runs(lengths: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous requests that draft the same number of mask rows. Zeros are skipped."""
    runs: list[tuple[int, int, int]] = []
    start = 0
    for i, n in enumerate(lengths.tolist()):
        if i == start:
            continue
        if int(n) != int(lengths[start]):
            if int(lengths[start]) > 0:
                runs.append((start, i, int(lengths[start])))
            start = i
    if len(lengths) > start and int(lengths[start]) > 0:
        runs.append((start, len(lengths), int(lengths[start])))
    return runs
