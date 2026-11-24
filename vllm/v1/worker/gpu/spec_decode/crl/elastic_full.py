# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-step varlen draft CUDA graphs, one per total query-row bucket.

The captured step is the whole draft: context KV, the bidirectional backbone,
candidate top-k, fused CRL (with a live FA3 schedule), the walk, and confidence.
Replay is selected by ``Q_pad`` alone. Padding rows carry ``req < 0`` or
``PAD_SLOT_ID`` and are not sampled.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.crl import elastic_plan
from vllm.v1.worker.gpu.spec_decode.crl.fused import (
    _crl_prep_varlen_kernel,
    _crl_score_varlen_kernel,
    _elastic_crl_kernel,
    _elastic_fix_qsl_kernel,
    _elastic_hidden_kernel,
    _elastic_scan_kernel,
    _elastic_scatter_kernel,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

logger = init_logger(__name__)

BUCKETS = (
    8,
    16,
    32,
    64,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1152,
    1280,
    1536,
)
MAX_NQ = 57  # 1 + (15 - 1) * 4


def bucket_shapes(q_pad: int) -> tuple[int, int]:
    """``D_pad`` candidate rows and ``C_pad`` CRL queries for a full bucket."""
    rmin = (q_pad + 15) // 16
    return q_pad - rmin, 4 * q_pad - 7 * rmin


def pick_bucket(q_tot: int, r_max: int) -> int | None:
    limit = r_max * 16
    for bucket in BUCKETS:
        if bucket >= q_tot and bucket <= limit:
            return bucket
    return None


class ElasticFullGraphs:
    def __init__(self, spec) -> None:
        self.spec = spec
        self.graphs: dict[int, dict] = {}
        self.hits = 0
        self.misses = 0
        self._has_gate = False
        self._async_active = False
        device = spec.device
        r = spec.max_num_reqs
        steps = spec.num_speculative_steps
        k = spec.top_k
        width = spec.model.model.config.hidden_size
        legal = [b for b in BUCKETS if b <= r * 16]
        if not legal:
            return
        max_d = max(bucket_shapes(b)[0] for b in legal)
        max_c = max(bucket_shapes(b)[1] for b in legal)
        self.r = r
        self.steps = steps
        self.k = k
        self.width = width
        self.q = torch.zeros(r, dtype=torch.int32, device=device)
        self.qcu = torch.zeros(r + 1, dtype=torch.int32, device=device)
        self.live = torch.zeros(1, dtype=torch.int32, device=device)
        self.limits = torch.zeros(r, dtype=torch.int32, device=device)
        self.hidden_idx = torch.zeros(max_d, dtype=torch.int64, device=device)
        self.cand_pos = torch.full((max_d,), -1, dtype=torch.int32, device=device)
        self.crl_req = torch.full((max_c,), -1, dtype=torch.int32, device=device)
        self.crl_qlocal = torch.zeros(max_c, dtype=torch.int32, device=device)
        self.row_cu = torch.zeros(r, dtype=torch.int32, device=device)
        self.crl_qcu = torch.zeros(r + 1, dtype=torch.int32, device=device)
        self.cand = torch.zeros(r, steps, k, dtype=torch.int64, device=device)
        self.unary = torch.full((r, steps, k), float("-inf"), device=device)
        self.table = torch.empty(r, steps, k, k, device=device)
        self.x1 = torch.empty(max_c, width, dtype=spec.dtype, device=device)
        self.residual = torch.empty(max_c, width, dtype=spec.dtype, device=device)
        self.crl_pos = torch.zeros(max_c, dtype=torch.int64, device=device)
        self.batch_pos = torch.zeros(
            spec.max_num_tokens, dtype=torch.int64, device=device
        )
        self.batch_qsl = torch.zeros(r + 1, dtype=torch.int32, device=device)
        self.batch_idx = torch.zeros(r, dtype=torch.int64, device=device)
        self.num_sampled = torch.zeros(r, dtype=torch.int32, device=device)
        self.num_rejected = torch.zeros(r, dtype=torch.int32, device=device)
        self.last_sampled = torch.zeros(r, dtype=torch.int64, device=device)
        self.next_prefill = torch.zeros(r, dtype=torch.int64, device=device)
        self.in_temp = torch.zeros(r, dtype=torch.float32, device=device)
        self.in_seeds = torch.zeros(r, dtype=torch.int64, device=device)
        mbt = int(
            getattr(spec.vllm_config.scheduler_config, "max_num_batched_tokens", 8192)
        )
        self.fixed_blocks = max(1, (mbt + int(spec.num_query_per_req) + 127) // 128)
        self.ctx = spec._num_graph_context_tokens(r)
        self._measured: list[tuple[int, float]] = []
        self._pre_staged = False
        self._has_gate = False
        self._async_active = False
        self._has_async = False
        self.live_arange = torch.arange(r, dtype=torch.int32, device=device)
        self.step_arange = torch.arange(steps, dtype=torch.int32, device=device)
        self.gate_req_ids = torch.full((r,), -1, dtype=torch.int64, device=device)
        self.gate_p_next = torch.zeros(1, dtype=torch.float32, device=device)
        self._ring_size = 3
        self.gate_req_cpu = [
            torch.empty(r, dtype=torch.int64, pin_memory=True)
            for _ in range(self._ring_size)
        ]
        self.gate_p_next_cpu = [
            torch.zeros(1, dtype=torch.float32, pin_memory=True)
            for _ in range(self._ring_size)
        ]
        self._h2d_step = 0
        self.uq_slot = torch.zeros(r, dtype=torch.int32, device=device)
        self.lam_out = torch.zeros(1, dtype=torch.float32, device=device)
        self._copy_stream = torch.cuda.Stream(device)
        self._copy_events = [torch.cuda.Event() for _ in range(self._ring_size)]
        self._async_gpu = torch.zeros(2 * r + 1, dtype=torch.int32, device=device)
        self._async_cpu = [
            torch.zeros(2 * r + 1, dtype=torch.int32, pin_memory=True)
            for _ in range(self._ring_size)
        ]
        self._async_np = [t.numpy() for t in self._async_cpu]
        self._ring_written = 0
        self._ring_ready = -1
        self._host_req_ids = np.full(r, -1, dtype=np.int64)
        self._host_new_cd = np.zeros(r, dtype=np.int32)
        self._host_d = np.full(r, 15, dtype=np.int32)
        self._host_slot_max_d = np.full(r, 15, dtype=np.int32)
        self._host_decode = np.ones(r, dtype=bool)
        self._host_lam = 0.0
        self._prev_q_est = 0

    def capture_all(self) -> None:
        spec = self.spec
        if not hasattr(self, "q"):
            return
        self.graphs.clear()
        gate = spec._load_anchor_gate()
        self._has_gate = (gate is not None) and (
            not os.environ.get("VLLM_ELASTIC_G", "")
        )
        self._warmup_gate_kernels()
        legal = [b for b in BUCKETS if b <= self.r * 16]
        self._measured = []
        pool = None
        targets: list[tuple[int, int, bool]] = [
            (q_pad, self.r, True) for q_pad in legal
        ]
        if self.r > 32:
            targets.extend((q_pad, 32, False) for q_pad in legal if q_pad <= 512)
        for q_pad, r_pad, is_full in targets:
            try:
                entry = self._build_entry(q_pad, r_pad)
                self._fill_dummy(q_pad, r_pad)
                torch.cuda.synchronize()
                self._step(entry)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._step(entry)
                pool = graph.pool()
                entry["graph"] = graph
                ms = self._time_graph(graph)
                entry["ms"] = ms
                self.graphs[(q_pad, r_pad)] = entry
                if is_full:
                    self.graphs[q_pad] = entry
                    self._measured.append((q_pad, ms))
                d_pad, c_pad = bucket_shapes(q_pad)
                logger.info(
                    "ELASTIC_FULL_GRAPH Q_pad=%d R_pad=%d D_pad=%d C_pad=%d ms=%.3f",
                    q_pad,
                    r_pad,
                    d_pad,
                    c_pad,
                    ms,
                )
            except Exception:
                logger.exception(
                    "ELASTIC_FULL_GRAPH capture failed Q_pad=%d R_pad=%d", q_pad, r_pad
                )
        if self._measured:
            self._publish_times()

    def _warmup_gate_kernels(self) -> None:
        from vllm.v1.worker.gpu.spec_decode.crl import elastic_plan
        from vllm.v1.worker.gpu.spec_decode.crl.fused import (
            _gpu_observe_kernel,
            _pava4_kernel,
            _stage_elastic_reqs_kernel,
            _traj_bind_state_kernel,
        )

        spec = self.spec
        dev = spec.device
        r = self.r
        raw = torch.ones((r, 4), dtype=torch.float32, device=dev)
        delta = torch.empty_like(raw)
        _pava4_kernel[(r,)](raw, delta)
        slots = torch.arange(r, dtype=torch.int64, device=dev)
        req = torch.arange(r, dtype=torch.int64, device=dev)
        out = torch.empty((r, 5), dtype=torch.float32, device=dev)
        spec._ensure_gpu_traj()
        _traj_bind_state_kernel[(1,)](
            slots,
            req,
            spec._gpu_req,
            spec._gpu_tau,
            spec._gpu_ema,
            spec._gpu_hit,
            spec._gpu_q,
            spec._gpu_steps,
            spec._gpu_d,
            out,
            self.live,
            r,
            HAS_LIVE=1,
            BLOCK=128,
        )
        conf = spec.draft_token_confidence_probs
        _gpu_observe_kernel[(1,)](
            slots,
            self.num_sampled,
            self.num_rejected,
            conf,
            conf.stride(0),
            spec._gpu_tau,
            spec._gpu_ema,
            spec._gpu_hit,
            spec._gpu_q,
            spec._gpu_steps,
            spec._gpu_d,
            r,
            BLOCK=128,
        )
        dec = torch.ones(r, dtype=torch.bool, device=dev)
        hit = torch.zeros(r, dtype=torch.bool, device=dev)
        pd = torch.zeros(r, dtype=torch.int32, device=dev)
        cold = torch.zeros(r, dtype=torch.bool, device=dev)
        p_next_t = torch.zeros(1, dtype=torch.float32, device=dev)
        elastic_plan.choose_total_graph(
            delta,
            dec,
            p_next_t,
            hit,
            pd,
            cold,
            slots,
            self.q,
            spec._gpu_d,
            self.uq_slot,
            self.lam_out,
            self.live,
            max_q_rows=min(64, r * 16),
        )
        if r + 1 <= 128:
            _stage_elastic_reqs_kernel[(1,)](
                self.batch_qsl,
                self.batch_qsl,
                self.batch_idx,
                self.batch_idx,
                self.num_sampled,
                self.num_sampled,
                self.num_rejected,
                self.num_rejected,
                self.last_sampled,
                self.last_sampled,
                self.next_prefill,
                self.next_prefill,
                self.in_temp,
                self.in_temp,
                self.in_seeds,
                self.in_seeds,
                r,
                r,
                BLOCK=128,
            )
        spec._gpu_req.fill_(-1)
        spec._gpu_tau.zero_()
        spec._gpu_ema.zero_()
        spec._gpu_hit.zero_()
        spec._gpu_q.fill_(0.5)
        spec._gpu_steps.zero_()
        spec._gpu_d.zero_()
        self.q.zero_()
        self.uq_slot.zero_()
        self.lam_out.zero_()
        torch.cuda.synchronize()

    def replay(
        self,
        q_lens,
        input_batch,
        num_reqs: int,
        num_target_tokens: int,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dummy_run: bool,
    ) -> bool:
        del dummy_run
        pre_staged = self._pre_staged
        self._pre_staged = False
        q_tot = int(q_lens.sum())
        bucket = pick_bucket(q_tot, self.r)
        entry = self.graphs.get(bucket) if bucket is not None else None
        if entry is None:
            self.misses += 1
            logger.info("ELASTIC_FULL_MISS q_tot=%d bucket=%s", q_tot, bucket)
            return False
        if not pre_staged:
            self._stage(
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
            )
        else:
            self.q.zero_()
            q_gpu = getattr(self.spec, "_elastic_q_gpu", None)
            if q_gpu is not None and q_gpu.shape[0] == num_reqs:
                self.q[:num_reqs].copy_(q_gpu)
                self.spec._elastic_q_gpu = None
            else:
                self.q[:num_reqs].copy_(
                    torch.from_numpy(np.asarray(q_lens, dtype=np.int32))
                )
        # VLLM_ELASTIC_EAGER=1 runs the captured step in Python so a real-batch
        # fault names the op. Replay is the steady-state path.
        if os.environ.get("VLLM_ELASTIC_EAGER") == "1":
            self._step(entry)
            torch.cuda.synchronize()
        else:
            entry["graph"].replay()
        self.hits += 1
        if self.hits <= 2 or self.hits % 256 == 0:
            logger.info(
                "ELASTIC_FULL_HIT hits=%d misses=%d q_pad=%d",
                self.hits,
                self.misses,
                bucket,
            )
        return True

    def replay_async(
        self,
        input_batch,
        num_reqs: int,
        num_target_tokens: int,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
    ) -> None:
        """Zero-sync elastic propose using double-buffered async D2H state."""
        spec = self.spec
        r_max = self.r
        while self._ring_ready + 1 < self._ring_written:
            nxt = (self._ring_ready + 1) % self._ring_size
            if self._copy_events[nxt].query():
                self._ring_ready += 1
            else:
                break
        if self._ring_ready >= 0:
            arr = self._async_np[self._ring_ready % self._ring_size]
            stale_uq = arr[:r_max]
            stale_d = arr[r_max : 2 * r_max]
            stale_lam = float(arr[2 * r_max :].view(np.float32)[0])
        else:
            stale_uq = None
            stale_d = None
            stale_lam = 0.0

        slots = np.asarray(input_batch.idx_mapping_np[:num_reqs], dtype=np.int64)
        req_ids = np.fromiter(
            (
                hash(input_batch.req_ids[i]) & 0x7FFFFFFFFFFFFFFF
                for i in range(num_reqs)
            ),
            dtype=np.int64,
            count=num_reqs,
        )
        computed = getattr(input_batch, "num_computed_prefill_tokens_np", None)
        prefill_len = getattr(input_batch, "prefill_len_np", None)
        qsl_np = getattr(input_batch, "query_start_loc_np", None)
        if computed is not None and prefill_len is not None:
            if qsl_np is not None:
                chunk = np.diff(qsl_np[: num_reqs + 1])
                prefill_np = (computed[:num_reqs] + chunk) < prefill_len[:num_reqs]
            else:
                prefill_np = (prefill_len[:num_reqs] - computed[:num_reqs]) > 0
        else:
            prefill_np = np.zeros(num_reqs, dtype=bool)
        p_next = spec._elastic_p_next(input_batch, num_reqs, prefill_np)

        dec_mask = ~prefill_np
        # Reset prefill slots so their first decode step is treated as fresh
        # (stale_uq stays 0 for 1-2 async steps after prefill).
        if prefill_np.any():
            p_slots = slots[prefill_np]
            self._host_req_ids[p_slots] = -1
            self._host_new_cd[p_slots] = 0
        d_slots = slots[dec_mask]
        d_rids = req_ids[dec_mask]
        d_new = self._host_req_ids[d_slots] != d_rids
        self._host_req_ids[d_slots] = d_rids
        self._host_new_cd[d_slots] = np.where(
            d_new, 2, np.maximum(self._host_new_cd[d_slots] - 1, 0)
        )
        fresh_mask = np.zeros(num_reqs, dtype=bool)
        fresh_mask[dec_mask] = d_new | (self._host_new_cd[d_slots] > 0)
        floor7 = elastic_plan.is_floor7()
        fresh_q = 16 if (not floor7 or num_reqs <= 8) else 8
        fresh_d = fresh_q - 1
        if stale_uq is not None:
            q_est = np.where(fresh_mask, fresh_q, stale_uq[slots])
            d_est = np.where(fresh_mask, fresh_d, stale_d[slots])
        else:
            q_est = np.full(num_reqs, fresh_q, dtype=np.int32)
            d_est = np.full(num_reqs, fresh_d, dtype=np.int32)
        if floor7:
            q_est = np.where(prefill_np, 0, np.maximum(q_est, 8)).astype(np.int32)
            d_est = np.where(prefill_np, 0, np.maximum(d_est, 7)).astype(np.int32)
        else:
            q_est = np.where(prefill_np, 0, q_est).astype(np.int32)
            d_est = np.where(prefill_np, 0, d_est).astype(np.int32)

        self._host_d[slots] = d_est
        self._host_decode[slots] = ~prefill_np
        self._host_lam = stale_lam
        self._async_active = True

        q_tot_est = int(q_est.sum())
        if q_tot_est == 0:
            self._host_slot_max_d[slots] = 0
            elastic_plan.publish_depths(
                slots, np.zeros(num_reqs, dtype=np.int32), stale_lam, r_max
            )
            spec._store_hidden_states(
                num_target_tokens, last_hidden_states, aux_hidden_states
            )
            spec.prepare_context_anchor(
                num_reqs, input_batch.query_start_loc, num_rejected
            )
            spec._precompute_context_kv(0, num_target_tokens)
            pad_tok = int(spec.model.lm_head.weight.shape[0]) - 1
            spec.draft_tokens[:num_reqs].fill_(pad_tok)
            spec._gpu_d.index_fill_(0, input_batch.idx_mapping[:num_reqs], 0)
            return

        growth = max(0, q_tot_est - self._prev_q_est)
        self._prev_q_est = q_tot_est
        base_q = 8 * int(dec_mask.sum()) if floor7 else 0
        # stale_uq is already uncapped (HAS_GU=1), so q_tot_est <= base_q means
        # zero requests wanted >8 rows on GPU even if num_dec grew (e.g. 31 -> 32).
        if q_tot_est >= r_max * 16 or (floor7 and q_tot_est <= base_q):
            headroom = 0
        elif growth > 0:
            headroom = min(128, growth + 32)
        elif q_tot_est > 700:
            headroom = 128
        else:
            headroom = 32 if q_tot_est <= 384 else 64
        target_q = max(base_q, min(q_tot_est + headroom, r_max * 16))
        bucket = pick_bucket(target_q, r_max)
        if floor7:
            extra_rows = max(0, bucket - base_q)
            step_max_d = (
                7
                if extra_rows < 4
                else (11 if extra_rows < 8 else self.steps)
            )
        else:
            step_max_d = self.steps
        slot_max_d = np.where(prefill_np, 0, step_max_d).astype(np.int32)
        self._host_slot_max_d[slots] = slot_max_d
        if spec.enable_adaptive_verification and elastic_plan.verify_mode() != "off":
            d_pub = slot_max_d
        else:
            d_pub = d_est
        elastic_plan.publish_depths(slots, d_pub, stale_lam, r_max)
        r_pad = (
            32
            if (
                r_max > 32
                and num_reqs <= 32
                and bucket <= 512
                and (bucket, 32) in self.graphs
            )
            else r_max
        )
        entry = self.graphs.get((bucket, r_pad)) or self.graphs.get(bucket)
        assert entry is not None
        elastic_plan.publish_draft_ms(entry.get("ms", elastic_plan.t_draft_ms(bucket)))

        h2d_i = self._h2d_step % self._ring_size
        self._h2d_step += 1
        req_buf = self.gate_req_cpu[h2d_i]
        pn_buf = self.gate_p_next_cpu[h2d_i]
        req_buf[:num_reqs].copy_(torch.from_numpy(req_ids))
        self.gate_req_ids[:num_reqs].copy_(req_buf[:num_reqs], non_blocking=True)
        pn_buf[0] = float(p_next)
        self.gate_p_next.copy_(pn_buf, non_blocking=True)

        self._stage(
            None,
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
            ctx_pad=entry["ctx_pad"],
        )
        if os.environ.get("VLLM_ELASTIC_EAGER") == "1":
            self._step(entry)
            torch.cuda.synchronize()
        else:
            entry["graph"].replay()
        self.hits += 1
        if os.environ.get("VLLM_ELASTIC_DIAG") == "1" and self.hits in (
            1,
            2,
            5,
            10,
            20,
            40,
        ):
            torch.cuda.synchronize()
            q_live = self.q[:num_reqs].cpu().numpy()
            uq_live = self.uq_slot[slots].cpu().numpy()
            logger.info(
                "ELASTIC_DIAG hit=%d n=%d q_pad=%d q_est=%d q_sum=%d uq_sum=%d lam=%.4f q=%s",
                self.hits,
                num_reqs,
                bucket,
                q_tot_est,
                int(q_live.sum()),
                int(uq_live.sum()),
                float(self.lam_out.item()),
                q_live[:8].tolist(),
            )
        if self.hits <= 2 or self.hits % 256 == 0:
            logger.info(
                "ELASTIC_FULL_HIT hits=%d misses=%d q_pad=%d",
                self.hits,
                self.misses,
                bucket,
            )

        if self._ring_written - self._ring_ready < self._ring_size:
            write_idx = self._ring_written % self._ring_size
            current_stream = torch.cuda.current_stream(spec.device)
            self._copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._copy_stream):
                self._async_cpu[write_idx].copy_(self._async_gpu, non_blocking=True)
                self._copy_events[write_idx].record()
            self._ring_written += 1

    def _publish_times(self) -> None:
        measured = sorted(self._measured)
        q0, m0 = measured[0]
        if len(measured) >= 2:
            q1, m1 = measured[1]
            slope = (m1 - m0) / max(q1 - q0, 1)
            at0 = min(m0, max(0.2, m0 - slope * q0))
        else:
            at0 = m0 * 0.5
        rows = [0.0] + [float(q) for q, _ in measured]
        ms = [float(at0)] + [float(t) for _, t in measured]
        elastic_plan.set_measured_draft(rows, ms)
        logger.info(
            "ELASTIC_DRAFT_MS %s",
            " ".join(f"{int(r)}:{t:.3f}" for r, t in zip(rows, ms)),
        )

    def _time_graph(self, graph: torch.cuda.CUDAGraph) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        samples.sort()
        return samples[len(samples) // 2]

    def _build_entry(self, q_pad: int, r_pad: int | None = None) -> dict:
        spec = self.spec
        r_pad = self.r if r_pad is None else r_pad
        spec.draft_max_seq_len = int(spec.max_model_len)
        rmin = min(r_pad, q_pad // 16)
        q_cu = torch.zeros(r_pad + 1, dtype=torch.int32)
        q_cu[1 : rmin + 1] = torch.arange(1, rmin + 1) * 16
        q_cu[rmin + 1 :] = q_pad
        upper = torch.zeros(r_pad, dtype=torch.int32)
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=q_pad,
            num_reqs=r_pad,
        )
        backbone = spec._build_attn_metadata(
            num_reqs=r_pad,
            batch_desc=desc,
            query_start_loc_np=q_cu.numpy(),
            seq_lens_cpu_upper_bound=upper,
            step=16,
            causal=spec._group_causal,
        )
        for meta in backbone.values():
            if hasattr(meta, "max_query_len"):
                meta.max_query_len = 16
        layer = spec.model.model.layers[-1].self_attn.attn
        base = backbone[layer.layer_name]
        crl_meta = copy.copy(base)
        crl_meta.query_start_loc = self.crl_qcu[: r_pad + 1]
        crl_meta.num_actual_tokens = bucket_shapes(q_pad)[1]
        crl_meta.max_query_len = MAX_NQ
        crl_meta.causal = False
        if getattr(crl_meta, "scheduler_metadata", None) is None:
            crl_meta.scheduler_metadata = torch.zeros(
                8, dtype=torch.int32, device=spec.device
            )
        builder = None
        for groups in spec.attn_groups:
            for group in groups:
                if layer.layer_name in group.layer_names:
                    builder = group.get_metadata_builder()
        return {
            "q_pad": q_pad,
            "r_pad": r_pad,
            "d_pad": bucket_shapes(q_pad)[0],
            "c_pad": bucket_shapes(q_pad)[1],
            "ctx_pad": min(self.ctx, max(256, q_pad)),
            "backbone": backbone,
            "crl_meta": crl_meta,
            "crl_builder": builder,
            "layer_name": layer.layer_name,
        }

    def _fill_dummy(self, q_pad: int, r_pad: int | None = None) -> None:
        spec = self.spec
        r = self.r if r_pad is None else r_pad
        rmin = min(r, q_pad // 16)
        self.q.zero_()
        self.q[:rmin] = 16
        rem = q_pad - rmin * 16
        if rem > 0 and rmin < r:
            self.q[rmin] = rem
        self.live.fill_(r)
        self.batch_pos[:r] = torch.arange(r, device=spec.device)
        self.batch_qsl[: r + 1] = torch.arange(r + 1, device=spec.device)
        self.batch_idx[:r] = torch.arange(r, device=spec.device)
        self.num_sampled.fill_(1)
        self.num_rejected.zero_()
        self.last_sampled.zero_()
        self.next_prefill.zero_()
        self.in_temp.zero_()
        self.in_seeds.zero_()
        for table in spec.block_tables.input_block_tables:
            table[: self.r].zero_()
        if spec._aux_staging is not None:
            spec._aux_staging.zero_()
        spec.hidden_states[: self.ctx].zero_()
        self.gate_req_ids[:r] = torch.arange(r, dtype=torch.int64, device=spec.device)
        self.gate_p_next.zero_()

    def pre_stage(
        self,
        input_batch,
        num_reqs: int,
        num_target_tokens: int,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
    ) -> None:
        if not hasattr(self, "q") or num_target_tokens > self.ctx:
            return
        self._stage(
            None,
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
        )
        self._pre_staged = True

    def _stage(
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
        ctx_pad: int | None = None,
    ) -> None:
        spec = self.spec
        ctx_lim = self.ctx if ctx_pad is None else ctx_pad
        if q_lens is not None:
            self.q.zero_()
            q_gpu = getattr(self.spec, "_elastic_q_gpu", None)
            if q_gpu is not None and q_gpu.shape[0] == num_reqs:
                self.q[:num_reqs].copy_(q_gpu)
                self.spec._elastic_q_gpu = None
            else:
                self.q[:num_reqs].copy_(
                    torch.from_numpy(np.asarray(q_lens, dtype=np.int32))
                )
        self.live.fill_(num_reqs)
        ntok = num_target_tokens
        self._copy_state(self.batch_pos, input_batch.positions, ntok)
        if self.r + 1 <= 128 and input_batch.query_start_loc.is_contiguous():
            from vllm.v1.worker.gpu.spec_decode.crl.fused import (
                _stage_elastic_reqs_kernel,
            )

            _stage_elastic_reqs_kernel[(1,)](
                input_batch.query_start_loc,
                self.batch_qsl,
                input_batch.idx_mapping,
                self.batch_idx,
                num_sampled,
                self.num_sampled,
                num_rejected,
                self.num_rejected,
                last_sampled.reshape(-1),
                self.last_sampled,
                next_prefill_tokens.reshape(-1),
                self.next_prefill,
                temperature.reshape(-1),
                self.in_temp,
                seeds.reshape(-1),
                self.in_seeds,
                num_reqs,
                self.r,
                BLOCK=128,
            )
        else:
            self._copy_state(self.batch_qsl, input_batch.query_start_loc, num_reqs + 1)
            self._copy_state(self.batch_idx, input_batch.idx_mapping, num_reqs)
            self._copy_state(self.num_sampled, num_sampled, num_reqs)
            self._copy_state(self.num_rejected, num_rejected, num_reqs)
            self._copy_state(self.last_sampled, last_sampled, None)
            self._copy_state(self.next_prefill, next_prefill_tokens, None)
            self._copy_state(self.in_temp, temperature, None)
            self._copy_state(self.in_seeds, seeds, None)
        if ntok > ctx_lim:
            self._prepare()
            spec._store_hidden_states(
                ntok, last_hidden_states, aux_hidden_states, num_graph_tokens=ctx_lim
            )
            spec._precompute_context_kv(ctx_lim, ntok)
        else:
            spec._store_hidden_states(
                ntok, last_hidden_states, aux_hidden_states, num_graph_tokens=ctx_lim
            )

    @staticmethod
    def _copy_state(dst: torch.Tensor, src: torch.Tensor, n: int | None) -> None:
        # last_sampled is [R, 1] and next_prefill is [lookahead, R]. The prep
        # kernel indexes both as base_ptr + req_slot, i.e. the first row.
        flat = src.reshape(-1)
        limit = dst.shape[0] if n is None else min(n, dst.shape[0])
        n = min(limit, flat.shape[0])
        view = flat[:n]
        if view.dtype != dst.dtype:
            if (
                view.element_size() == dst.element_size()
                and not view.dtype.is_floating_point
                and not dst.dtype.is_floating_point
            ):
                view = view.view(dst.dtype)
            else:
                view = view.to(dtype=dst.dtype)
        dst[:n].copy_(view, non_blocking=True)

    def _prepare(self, r_pad: int | None = None) -> None:
        spec = self.spec
        r_pad = self.r if r_pad is None else r_pad
        if spec._query_cu_gpu is None:
            spec._query_cu_gpu = self.qcu
            spec._pos_in_block = torch.empty(
                spec.max_num_tokens, dtype=torch.int32, device=spec.device
            )
        for i, gid in enumerate(spec.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                spec.input_buffers,
                spec.block_tables.slot_mappings[gid],
                spec.context_positions,
                spec._context_slot_mappings[i],
                spec.sample_indices,
                spec.sample_pos,
                spec.sample_idx_mapping,
                spec.temperature,
                spec.seeds,
                spec.input_buffers,  # unused when the batch_* overrides are set
                self.num_sampled,
                self.num_rejected,
                self.last_sampled,
                self.next_prefill,
                self.in_temp,
                self.in_seeds,
                spec.block_tables.input_block_tables[gid],
                spec.block_tables.block_sizes[gid],
                spec.block_tables.cp_rank,
                spec.dcp_size,
                spec.block_tables.cp_interleave,
                spec.parallel_drafting_token_id,
                16,
                spec.num_speculative_steps,
                spec.max_num_reqs,
                spec.max_num_tokens,
                spec.max_model_len,
                spec.sample_from_anchor,
                r_pad,
                spec.group_size,
                query_cu=self.qcu,
                pos_in_block=spec._pos_in_block,
                live_reqs=self.live,
                fixed_blocks=self.fixed_blocks,
                batch_positions=self.batch_pos,
                batch_query_start_loc=self.batch_qsl,
                batch_idx_mapping=self.batch_idx,
            )

    def _gate_in_graph(
        self, q_pad: int, r_pad: int, gate: dict[str, torch.Tensor]
    ) -> None:
        from vllm.v1.worker.gpu.spec_decode.crl.fused import _pava4_kernel

        spec = self.spec
        r = r_pad
        ends = (
            (self.batch_qsl[1 : r + 1] - self.num_rejected[:r] - 1)
            .to(torch.long)
            .clamp(0, spec.hidden_states.shape[0] - 1)
        )
        projected = spec.hidden_states.index_select(0, ends)
        context = spec.model.model.hidden_norm(projected).reshape(r, -1)
        slots = self.batch_idx[:r].to(torch.long).clamp(0, self.r - 1)
        vocab_lim = int(spec.model.lm_head.weight.shape[0]) - 1
        tok_ids = (
            self.last_sampled[slots].reshape(-1).to(torch.long).clamp(0, vocab_lim)
        )
        embed = spec.model.embed_input_ids(tok_ids).reshape(r, -1)
        state = spec._gpu_bind_state_dev(slots, self.gate_req_ids[:r], live_t=self.live)
        ctx = context.float()
        emb = embed.float()
        h_s = torch.nn.functional.silu(state @ gate["W_s"].T + gate["b_s"])
        hidden = torch.nn.functional.silu(
            ctx @ gate["W_c"].T + emb @ gate["W_e"].T + h_s @ gate["W_su"].T
        )
        score = torch.cat((hidden, h_s), dim=-1) @ gate["W_o"].T + gate["b_o"]
        raw = (gate["C"] * torch.sigmoid(score)).contiguous()
        if elastic_plan.is_floor7():
            delta = raw
        else:
            delta = torch.empty_like(raw)
            _pava4_kernel[(r,)](raw, delta)
        live_mask = self.live_arange[:r] < self.live
        decode = (self.num_sampled[:r] > 0) & live_mask
        hit = spec._gpu_hit[slots] > 0.5
        prev_d = spec._gpu_d[slots]
        cold = spec._gpu_steps[slots] == 0
        elastic_plan.choose_total_graph(
            delta,
            decode,
            self.gate_p_next,
            hit,
            prev_d,
            cold,
            slots,
            self.q[:r],
            spec._gpu_d,
            self.uq_slot,
            self.lam_out,
            self.live,
            max_q_rows=q_pad,
        )

    def _step(self, entry: dict) -> None:
        spec = self.spec
        q_pad = entry["q_pad"]
        r_pad = entry["r_pad"]
        d_pad = entry["d_pad"]
        c_pad = entry["c_pad"]
        ctx_pad = entry["ctx_pad"]
        if spec._aux_staging is not None:
            spec.hidden_states[:ctx_pad].copy_(
                spec.model.combine_hidden_states(spec._aux_staging[:ctx_pad])
            )
        if self._has_gate:
            gate = spec._load_anchor_gate()
            assert gate is not None
            self._gate_in_graph(q_pad, r_pad, gate)
        # A Python scalar store is a CPU→CUDA copy, which capture rejects.
        self.qcu[:1].zero_()
        torch.cumsum(self.q[:r_pad], dim=0, out=self.qcu[1 : r_pad + 1])
        spec.context_positions[:ctx_pad].zero_()
        spec._context_slot_mappings.fill_(PAD_SLOT_ID)
        self._prepare(r_pad)
        _elastic_fix_qsl_kernel[(1,)](
            spec.input_buffers.query_start_loc,
            spec.input_buffers.seq_lens,
            self.qcu,
            self.live,
            spec.input_buffers.input_ids,
            spec.input_buffers.positions,
            spec.input_buffers.is_padding,
            spec._pos_in_block,
            spec.block_tables.slot_mappings,
            spec.block_tables.slot_mappings.stride(0),
            num_groups=int(spec.block_tables.slot_mappings.shape[0]),
            q_pad=q_pad,
            PAD_SLOT_ID=PAD_SLOT_ID,
            R=r_pad,
            BLOCK=128,
        )
        spec._precompute_context_kv(0, ctx_pad)
        spec._update_draft_decode_metadata(entry["backbone"], r_pad)
        spec._set_conv_positions(spec._pos_in_block[:q_pad])
        slots = build_slot_mappings_by_layer(
            spec.block_tables.slot_mappings[:, :q_pad],
            spec.kv_cache_config,
        )
        hidden = spec._run_model(
            q_pad,
            entry["backbone"],
            slots,
            None,
            CUDAGraphMode.NONE,
            skip_compiled=False,
        )
        ka = int(spec.model.model.crl.pred_topk)
        _elastic_scan_kernel[(1,)](
            self.q,
            self.limits,
            self.row_cu,
            self.crl_qcu,
            R=r_pad,
            KA=ka,
            C_PAD=c_pad,
            MAX_NQ=MAX_NQ,
        )
        _elastic_hidden_kernel[(d_pad,)](
            spec.sample_indices.view(self.r, self.steps)[:r_pad],
            self.limits,
            self.row_cu,
            self.hidden_idx[:d_pad],
            self.cand_pos[:d_pad],
            R=r_pad,
            STEPS=self.steps,
            D_PAD=d_pad,
        )
        _elastic_crl_kernel[(c_pad,)](
            self.limits,
            self.crl_qcu,
            self.crl_req[:c_pad],
            self.crl_qlocal[:c_pad],
            R=r_pad,
            KA=ka,
            C_PAD=c_pad,
        )
        gathered = hidden.index_select(0, self.hidden_idx[:d_pad].clamp(0, q_pad - 1))
        ids, logits = spec.model.compute_candidates(gathered)
        cand = self.cand[:r_pad]
        unary = self.unary[:r_pad]
        table = self.table[:r_pad]
        cand.fill_(0)
        unary.fill_(float("-inf"))
        _elastic_scatter_kernel[(d_pad,)](
            ids,
            logits.float(),
            self.cand_pos[:d_pad],
            cand.view(-1, self.k),
            unary.view(-1, self.k),
            K=self.k,
        )
        table.copy_(unary[:, :, None, :].expand_as(table))
        self._live_hidden = hidden
        self._score_hidden(entry)
        pad_tok = int(spec.model.lm_head.weight.shape[0]) - 1
        spec.draft_tokens[:r_pad].fill_(pad_tok)
        spec.candidate_sampler.sample(
            cand,
            table,
            r_pad,
            spec.sample_pos,
            spec.sample_idx_mapping,
            spec.temperature,
            spec.seeds,
            spec.draft_tokens,
            spec.draft_logits,
            spec.use_fp64_gumbel,
            spec.draft_logits_num_shards,
            spec.draft_logits_shard_idx,
            row_limits=self.limits[:r_pad],
            pad_token=pad_tok,
        )
        if spec.enable_adaptive_verification:
            spec._crl_attn_metadata = entry["backbone"]
            conf = spec.draft_token_confidence_probs[:r_pad]
            conf.fill_(1)
            spec._predict_confidence(cand, r_pad, r_pad * self.steps)
            conf.masked_fill_(~torch.isfinite(conf), 1)
            conf.masked_fill_(
                self.step_arange[None, :] >= self.limits[:r_pad, None], 0.0
            )
        if self._has_gate:
            r_max = self.r
            self._async_gpu[:r_max].copy_(self.uq_slot)
            self._async_gpu[r_max : 2 * r_max].copy_(spec._gpu_d[:r_max])
            self._async_gpu[2 * r_max :].copy_(self.lam_out.view(torch.int32))

    def _score_hidden(self, entry: dict) -> None:
        spec = self.spec
        model = spec.model.model
        crl = model.crl
        c_pad = entry["c_pad"]
        r_pad = entry["r_pad"]
        hidden = self._live_hidden
        packed_h = hidden.index_select(
            0, self.hidden_idx[: entry["d_pad"]].clamp(0, entry["q_pad"] - 1)
        )
        block = 1
        while block < self.width:
            block *= 2
        anchors = spec.input_buffers.input_ids.index_select(
            0,
            self.qcu[:r_pad].long().clamp(0, spec.input_buffers.input_ids.shape[0] - 1),
        )
        sample_pos = spec.sample_pos.view(self.r, self.steps)[:r_pad]
        _crl_prep_varlen_kernel[(c_pad,)](
            packed_h,
            packed_h.stride(0),
            anchors,
            self.cand,
            sample_pos,
            crl.emb_table,
            crl.norm1.weight,
            self.x1[:c_pad],
            self.residual[:c_pad],
            self.crl_pos[:c_pad],
            self.crl_req[:c_pad],
            self.crl_qlocal[:c_pad],
            self.row_cu,
            crl.norm1.variance_epsilon,
            STEPS=self.steps,
            top_k=self.k,
            ka=int(crl.pred_topk),
            D=self.width,
            BLOCK=block,
        )
        q = crl.q_proj(self.x1[:c_pad])
        qh = q.view(-1, crl.head_dim)
        qn = torch.empty_like(qh)
        ops.rms_norm(qn, qh, crl.q_norm.weight, crl.q_norm.variance_epsilon)
        q = qn.view(c_pad, -1)
        cache = model._rope_cos_sin_cache
        if cache.dtype != q.dtype:
            cache = cache.to(dtype=q.dtype)
        ops.rotary_embedding(
            self.crl_pos[:c_pad],
            q,
            None,
            model._rope_head_size,
            cache,
            model._rope_is_neox,
        )
        layer = model.layers[-1].self_attn.attn
        if entry["crl_builder"] is not None:
            entry["crl_builder"].update_draft_decode_metadata(entry["crl_meta"])
        heads = q.view(c_pad, -1, crl.head_dim)
        out = torch.empty_like(heads)
        layer.impl.forward(
            layer, heads, heads, heads, layer.kv_cache, entry["crl_meta"], out
        )
        x2 = crl.o_proj(out.view(c_pad, -1))
        ops.fused_add_rms_norm(
            x2, self.residual[:c_pad], crl.norm2.weight, crl.norm2.variance_epsilon
        )
        down = crl.down_proj(
            crl.act(torch.nn.functional.linear(x2, crl.gate_up_weight))
        )
        ops.fused_add_rms_norm(
            down, self.residual[:c_pad], model.norm.weight, model.norm.variance_epsilon
        )
        null = spec._conf_mode == "null" and spec.enable_adaptive_verification
        if null and spec._null_delta is not None:
            spec._null_delta[:r_pad].zero_()
            delta = spec._null_delta
        else:
            delta = self.table
        _crl_score_varlen_kernel[(c_pad,)](
            down,
            self.cand,
            spec.model.lm_head.weight,
            self.table,
            spec._null_w if null else self.table,
            delta,
            self.crl_req[:c_pad],
            self.crl_qlocal[:c_pad],
            STEPS=self.steps,
            top_k=self.k,
            ka=int(crl.pred_topk),
            D=self.width,
            BLOCK_D=min(1024, block),
            NULL=null,
        )
