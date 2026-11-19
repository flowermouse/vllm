# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Steady-state step breakdown. Off unless ``VLLM_CRL_STEP_PROFILE`` is set.

CUDA events time the target forward, the sampler, and the drafter. The token
counts are the ones the forward was dispatched with (kept tokens, and the
capture bucket they were padded to). One synchronize when the window closes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def enabled() -> bool:
    return bool(os.environ.get("VLLM_CRL_STEP_PROFILE"))


def get(runner: object) -> StepProbe | None:
    if not enabled():
        return None
    probe = getattr(runner, "_crl_step_probe", None)
    if probe is None:
        probe = StepProbe()
        runner._crl_step_probe = probe  # type: ignore[attr-defined]
    return probe


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@dataclass
class _Row:
    tokens: int
    padded: int
    cg: str
    reqs: int
    drafts: int
    budget_ms: float
    fwd_s: torch.cuda.Event
    fwd_e: torch.cuda.Event
    samp_s: torch.cuda.Event
    samp_e: torch.cuda.Event
    draft_s: torch.cuda.Event
    draft_e: torch.cuda.Event
    has_draft: bool = False
    has_sample: bool = False


class StepProbe:
    def __init__(self) -> None:
        self._skip = int(os.environ.get("VLLM_CRL_STEP_PROFILE_SKIP", "8"))
        self._min_reqs = int(os.environ.get("VLLM_CRL_STEP_PROFILE_MIN_REQS", "1"))
        self._left = int(os.environ["VLLM_CRL_STEP_PROFILE"])
        self._seen = 0
        self._rows: list[_Row] = []
        self._cur: _Row | None = None
        self._done = False
        self._cpu_s = 0.0

    def add_cpu(self, seconds: float) -> None:
        """Host time of budget selection and compaction, before the row exists."""
        self._cpu_s += seconds

    def begin_forward(
        self,
        *,
        tokens: int,
        padded: int,
        cg: str,
        reqs: int,
        drafts: int,
        prefill: bool,
    ) -> None:
        budget_ms = self._cpu_s * 1e3
        self._cpu_s = 0.0
        if self._done or prefill or reqs < self._min_reqs:
            self._cur = None
            return
        self._seen += 1
        if self._seen <= self._skip or self._left <= 0:
            self._cur = None
            return
        self._left -= 1
        self._cur = _Row(
            tokens,
            padded,
            cg,
            reqs,
            drafts,
            budget_ms,
            _event(),
            _event(),
            _event(),
            _event(),
            _event(),
            _event(),
        )
        self._cur.fwd_s.record()

    def end_forward(self) -> None:
        if self._cur is not None:
            self._cur.fwd_e.record()

    def begin_sample(self) -> None:
        if self._cur is not None:
            self._cur.samp_s.record()

    def end_sample(self) -> None:
        if self._cur is not None:
            self._cur.has_sample = True
            self._cur.samp_e.record()

    def begin_draft(self) -> None:
        if self._cur is not None:
            self._cur.draft_s.record()

    def end_draft(self) -> None:
        row = self._cur
        if row is None:
            return
        row.has_draft = True
        row.draft_e.record()
        self._rows.append(row)
        self._cur = None
        if self._left == 0:
            self._finish()

    def _finish(self) -> None:
        self._done = True
        rows = self._rows
        if not rows:
            return
        rows[-1].draft_e.synchronize()

        def ms(a: torch.cuda.Event, b: torch.cuda.Event) -> float:
            return a.elapsed_time(b)

        fwd = [ms(r.fwd_s, r.fwd_e) for r in rows]
        samp = [ms(r.samp_s, r.samp_e) for r in rows if r.has_sample]
        draft = [ms(r.draft_s, r.draft_e) for r in rows if r.has_draft]
        n = len(rows)
        logger.info(
            "CRL step probe n=%d tokens=%d..%d padded=%d..%d cg=%s reqs=%d drafts=%d..%d "
            "budget_cpu_ms=%.3f forward_ms=%.3f sample_ms=%.3f draft_ms=%.3f",
            n,
            min(r.tokens for r in rows),
            max(r.tokens for r in rows),
            min(r.padded for r in rows),
            max(r.padded for r in rows),
            rows[n // 2].cg,
            rows[n // 2].reqs,
            min(r.drafts for r in rows),
            max(r.drafts for r in rows),
            sorted(r.budget_ms for r in rows)[n // 2],
            sorted(fwd)[n // 2],
            sorted(samp)[len(samp) // 2] if samp else float("nan"),
            sorted(draft)[len(draft) // 2] if draft else float("nan"),
        )
