# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-level budget for Ours elastic drafting.

The first level picks a total tier count before the draft forward. The second
level, in adaptive verification, cuts how many of those drafts are verified.
Both read the pure-decode opportunity cost from the latest draft step.
"""

from __future__ import annotations

import os

import numpy as np

# Piecewise draft time, query rows -> ms. Flat through the bandwidth region
# (<=512) and steeper once the H200 is compute bound. Points are measured
# medians (c32 256/384/512, c96 576/1152/1536); the rest are linear fills.
_DRAFT_ROWS = np.asarray([0, 64, 128, 256, 384, 512, 768, 1024, 1536], dtype=np.float64)
# This card: c32 192 tok 3.46 ms, 384 tok 4.13 ms, 512 tok 4.12 ms;
# c96 576 tok 5.59 ms, 1152 tok 8.92 ms, 1536 tok 11.73 ms.
_DRAFT_MS = np.asarray(
    [2.20, 2.60, 3.00, 3.60, 4.13, 4.12, 6.04, 7.96, 11.73], dtype=np.float64
)
# A real H200 verify of a short batch is several milliseconds. 0.4 ms made the
# opportunity cost treat verification as free and collapse the chosen tier.
_VERIFY_FLOOR_MS = 5.4
_VERIFY_PER_TOKEN_MS = 0.008
# Share of a drafted tier that Null actually forwards. Fixed from the seed-0
# calibration set; not tuned online.
ALPHA_PASS = 0.80

_depths: np.ndarray | None = None
_lambda: float | None = None
_last_draft_ms: float | None = None
_active = False
_verify_cost: np.ndarray | None = None

TIER_SLOTS = np.asarray([3, 4, 4, 4], dtype=np.int32)
# Query rows added by tiers 0..3. Tier 0 is an anchor plus 3 drafts.
TIER_QUERY = np.asarray([4, 4, 4, 4], dtype=np.int32)
_DEPTH_OF_G = np.asarray([0, 3, 7, 11, 15], dtype=np.int32)


def verify_mode() -> str:
    """``lambda`` is the opportunity-cost cut, ``ratio`` the old throughput ratio, ``off`` keeps every drafted token."""
    return os.environ.get("VLLM_ELASTIC_VERIFY", "lambda")


def publish_verify_cost(table: np.ndarray) -> None:
    global _verify_cost
    _verify_cost = np.asarray(table, dtype=np.float64)
    _sync_gpu_tables()


def verify_cost_ready() -> bool:
    """False while CUDA-graph capture runs, before adaptive verification publishes its table."""
    return _verify_cost is not None


def drafted_counts(slot_ids: np.ndarray) -> np.ndarray | None:
    """Draft tokens actually written for these request slots. ``None`` when elastic is off.

    A slot with ``g_i = 0`` is 0, so the scheduler must not keep a placeholder.
    """
    if not _active or _depths is None:
        return None
    return _depths[np.asarray(slot_ids, dtype=np.int64)]


def t_draft_ms(query_rows: int) -> float:
    rows = float(max(query_rows, 0))
    return float(np.interp(rows, _DRAFT_ROWS, _DRAFT_MS))


def t_verify_ms(n_tokens: int) -> float:
    n = max(int(n_tokens), 0)
    if _verify_cost is not None and n < len(_verify_cost):
        return float(_verify_cost[n])
    return _VERIFY_FLOOR_MS + _VERIFY_PER_TOKEN_MS * n


def pava_decreasing(rows: np.ndarray) -> np.ndarray:
    """Pool adjacent increases so each row is monotone non-increasing. ``rows`` is ``[R, 4]``."""
    out = np.empty_like(rows, dtype=np.float64)
    for i, values in enumerate(rows):
        y = [float(v) for v in values]
        w = [1.0] * len(y)
        j = 0
        while j < len(y) - 1:
            if y[j] + 1e-12 >= y[j + 1]:
                j += 1
                continue
            total_w = w[j] + w[j + 1]
            y[j] = (y[j] * w[j] + y[j + 1] * w[j + 1]) / total_w
            w[j] = total_w
            del y[j + 1]
            del w[j + 1]
            while j > 0 and y[j - 1] + 1e-12 < y[j]:
                total_w = w[j - 1] + w[j]
                y[j - 1] = (y[j - 1] * w[j - 1] + y[j] * w[j]) / total_w
                w[j - 1] = total_w
                del y[j]
                del w[j]
                j -= 1
        flat: list[float] = []
        for value, weight in zip(y, w):
            flat.extend([value] * int(round(weight)))
        out[i] = flat
    return out


def choose_total(
    delta: np.ndarray,
    decode: np.ndarray,
    p_next: int,
    hit_cap: np.ndarray | None = None,
    prev_d: np.ndarray | None = None,
    cold: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return query rows ``q_i = 4 g_i`` and the pure-decode opportunity cost.

    ``delta`` is ``[R, 4]`` expected accepts per tier. Non-decode rows are ignored.
    With Null on, a tier is charged only for the tokens expected to survive it.
    ``hit_cap`` promotes a request that filled a short draft, so a censored
    length cannot pin the next step.
    """
    r = int(np.asarray(decode).sum())
    option = verify_mode() != "off"
    items: list[tuple[float, int, int, float, int]] = []
    for i in np.flatnonzero(np.asarray(decode)):
        for tier in range(4):
            value = float(delta[i, tier])
            if value <= 0:
                continue
            cap = int(TIER_SLOTS[tier])
            verified = float(cap if not option else min(cap, value / ALPHA_PASS))
            items.append((value, int(i), tier, verified, int(TIER_QUERY[tier])))
    items.sort(key=lambda it: it[0], reverse=True)
    prefix = [(0, 0.0, 0.0, 0)]
    accepted = 0.0
    verified = 0.0
    rows = 0
    for value, _req, _tier, ver, qrow in items:
        accepted += value
        verified += ver
        rows += qrow
        prefix.append((prefix[-1][0] + 1, accepted, verified, rows))
    best = -1.0
    lam = 0.0
    for _g, y, d_hat, qrows in prefix:
        rate = (r + y) / (t_draft_ms(qrows) + t_verify_ms(int(round(r + d_hat))))
        if rate > best:
            best = rate
            lam = rate
    best_u = -1e300
    g_tot = 0
    for g, y, d_hat, qrows in prefix:
        utility = y - lam * (
            t_draft_ms(qrows) + t_verify_ms(int(round(p_next + r + d_hat)))
        )
        if utility > best_u:
            best_u = utility
            g_tot = g
    g = np.zeros(delta.shape[0], dtype=np.int32)
    for _value, req, _tier, _ver, _qrow in items[:g_tot]:
        g[req] += 1
    if hit_cap is not None and prev_d is not None:
        for i in np.flatnonzero(np.asarray(decode)):
            if not hit_cap[i] or not (0 < int(prev_d[i]) < 15):
                continue
            # A full hit below 11 means the chain was censored. Jump to at
            # least d=11; a full hit at 11 goes to the full block.
            floor = {3: 3, 7: 3, 11: 4}[int(prev_d[i])]
            g[i] = max(int(g[i]), floor)
    if cold is not None:
        for i in np.flatnonzero(np.asarray(decode) & np.asarray(cold)):
            if float(delta[i, 0]) > 0:
                g[i] = max(int(g[i]), 2)
    return (4 * g).astype(np.int32), float(lam)


# Per-slot trajectory. A slot is reset when a different request lands in it.
_TRAJ_N = 0
_traj_req = np.zeros(0, dtype=np.int64)
_traj_tau = np.zeros(0, dtype=np.float32)
_traj_ema = np.zeros(0, dtype=np.float32)
_traj_hit = np.zeros(0, dtype=np.float32)
_traj_q = np.zeros(0, dtype=np.float32)
_traj_steps = np.zeros(0, dtype=np.int32)
_traj_d = np.zeros(0, dtype=np.int32)


def _ensure_traj(n_slots: int) -> None:
    global \
        _TRAJ_N, \
        _traj_req, \
        _traj_tau, \
        _traj_ema, \
        _traj_hit, \
        _traj_q, \
        _traj_steps, \
        _traj_d
    if n_slots <= _TRAJ_N:
        return

    def grow(old, fill):
        out = np.full(n_slots, fill, dtype=old.dtype)
        if len(old):
            out[: len(old)] = old
        return out

    _traj_req = grow(_traj_req, -1)
    _traj_tau = grow(_traj_tau, 0)
    _traj_ema = grow(_traj_ema, 0)
    _traj_hit = grow(_traj_hit, 0)
    _traj_q = grow(_traj_q, 0.5)
    _traj_steps = grow(_traj_steps, 0)
    _traj_d = grow(_traj_d, 0)
    _TRAJ_N = n_slots


def reset_slot(slot: int) -> None:
    _ensure_traj(slot + 1)
    _traj_req[slot] = -1
    _traj_tau[slot] = 0
    _traj_ema[slot] = 0
    _traj_hit[slot] = 0
    _traj_q[slot] = 0.5
    _traj_steps[slot] = 0
    _traj_d[slot] = 0


def bind_requests(slot_ids: np.ndarray, req_ids: np.ndarray) -> None:
    """Drop a slot's history when a new request occupies it."""
    if len(slot_ids) == 0:
        return
    _ensure_traj(int(np.max(slot_ids)) + 1)
    for slot, req in zip(slot_ids.tolist(), req_ids.tolist()):
        if int(_traj_req[slot]) != int(req):
            reset_slot(int(slot))
            _traj_req[slot] = int(req)


def trajectory_features(slot_ids: np.ndarray) -> np.ndarray:
    """State for the step about to be drafted. Shape ``[R, 5]``."""
    out = np.zeros((len(slot_ids), 5), dtype=np.float32)
    if len(slot_ids) == 0:
        return out
    _ensure_traj(int(np.max(slot_ids)) + 1)
    for i, slot in enumerate(slot_ids.tolist()):
        out[i, 0] = float(_traj_tau[slot]) / 15.0
        out[i, 1] = float(_traj_ema[slot]) / 15.0
        out[i, 2] = float(_traj_hit[slot])
        out[i, 3] = float(_traj_q[slot])
        out[i, 4] = 1.0 if int(_traj_steps[slot]) == 0 else 0.0
    return out


def trajectory_side(slot_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_traj(int(np.max(slot_ids)) + 1 if len(slot_ids) else 1)
    slots = slot_ids.tolist()
    hit = np.asarray([_traj_hit[s] > 0.5 for s in slots], dtype=bool)
    prev = np.asarray([_traj_d[s] for s in slots], dtype=np.int32)
    cold = np.asarray([_traj_steps[s] == 0 for s in slots], dtype=bool)
    return hit, prev, cold


def observe_accepts(
    slot_ids: np.ndarray,
    accepted: np.ndarray,
    rejected: np.ndarray,
    confidence: np.ndarray | None,
) -> None:
    """Fold one finished draft step in. ``accepted`` excludes the bonus token."""
    if len(slot_ids) == 0:
        return
    _ensure_traj(int(np.max(slot_ids)) + 1)
    for i, slot in enumerate(slot_ids.tolist()):
        depth = int(_traj_d[slot])
        if depth <= 0:
            continue
        tau = float(accepted[i])
        # A full hit of a short draft must not pull a long-chain EMA down.
        if tau >= depth and float(_traj_ema[slot]) > tau:
            pass
        else:
            _traj_ema[slot] = 0.7 * float(_traj_ema[slot]) + 0.3 * tau
        _traj_tau[slot] = tau
        _traj_hit[slot] = 1.0 if tau >= depth and float(rejected[i]) == 0 else 0.0
        if confidence is not None:
            row = np.asarray(confidence[i, :depth], dtype=np.float32)
            finite = row[np.isfinite(row)]
            _traj_q[slot] = float(finite.mean()) if len(finite) else 0.5
        _traj_steps[slot] = int(_traj_steps[slot]) + 1


def publish(
    slot_ids: np.ndarray, query_rows: np.ndarray, lam: float, max_reqs: int
) -> None:
    """Remember each request's drafted depth and the opportunity cost for the next verify."""
    depths = np.where(query_rows > 0, query_rows - 1, 0).astype(np.int32)
    publish_depths(slot_ids, depths, lam, max_reqs)


def publish_depths(
    slot_ids: np.ndarray, depths: np.ndarray, lam: float, max_reqs: int
) -> None:
    """Publish drafted depths directly (used by async double-buffered elastic gate)."""
    global _depths, _lambda, _active
    if _depths is None or len(_depths) < max_reqs:
        _depths = np.full(max_reqs, 15, dtype=np.int32)
    depths = np.asarray(depths, dtype=np.int32)
    _depths[np.asarray(slot_ids, dtype=np.int64)] = depths
    _lambda = float(lam)
    _active = True


def cap_scheduled(
    slot_ids: np.ndarray, scheduled: np.ndarray
) -> tuple[np.ndarray, str]:
    """Cap scheduled drafts at the depth the last elastic step actually wrote."""
    if not _active or _depths is None:
        return scheduled, ""
    capped = np.minimum(scheduled, _depths[np.asarray(slot_ids, dtype=np.int64)])
    return capped.astype(np.int32, copy=False), verify_mode()


def lambda_star() -> float | None:
    return _lambda


def publish_draft_ms(ms: float) -> None:
    global _last_draft_ms
    _last_draft_ms = float(ms)


def last_draft_ms() -> float | None:
    return _last_draft_ms


def verify_cut(
    scores: np.ndarray, non_draft_tokens: int, verify_cost: np.ndarray
) -> int:
    """Pick how many of the already-drafted tokens to verify. ``scores`` is best-first."""
    lam = 0.0 if _lambda is None else _lambda
    best_b = 0
    best_u = -1e300
    accepted = 0.0
    last = len(verify_cost) - 1
    for b in range(len(scores) + 1):
        if b:
            accepted += float(scores[b - 1])
        cost = float(verify_cost[min(non_draft_tokens + b, last)])
        utility = accepted - lam * cost
        if utility > best_u:
            best_u = utility
            best_b = b
    return best_b


def _self_test() -> None:
    inverted = np.asarray([[1.0, 3.0, 2.0, 0.5]])
    pooled = pava_decreasing(inverted)[0]
    assert pooled[0] >= pooled[1] >= pooled[2] >= pooled[3]
    assert abs(pooled[0] + pooled[1] - 4.0) < 1e-6
    # High-accept requests keep a longer prefix than low-accept ones when verify is expensive.
    delta = np.asarray(
        [
            [2.5, 2.0, 1.5, 1.0],
            [0.2, 0.1, 0.05, 0.01],
        ]
    )
    q, lam = choose_total(delta, np.asarray([True, True]), p_next=0)
    assert q[0] > q[1]  # the long chain outbids the short one
    assert q[0] == 16  # its four tiers all clear the option price
    assert lam > 0
    dead = np.asarray([[0.0, 0.0, 0.0, 0.0]])
    q_dead, _ = choose_total(dead, np.asarray([True]), p_next=0)
    assert q_dead[0] == 0
    # A prefill row stays at zero even if its numbers are large.
    q2, _ = choose_total(delta, np.asarray([False, True]), p_next=10_000)
    assert q2[0] == 0
    # A full hit on d=3 cannot keep the next step at d=3.
    short = np.asarray([[0.05, 0.0, 0.0, 0.0]])
    q3, _ = choose_total(
        short,
        np.asarray([True]),
        p_next=0,
        hit_cap=np.asarray([True]),
        prev_d=np.asarray([3]),
    )
    assert q3[0] >= 8
    publish(np.asarray([3, 4]), np.asarray([0, 8]), lam, 8)
    assert drafted_counts(np.asarray([3, 4])).tolist() == [0, 7]
    bind_requests(np.asarray([0]), np.asarray([7]))
    assert trajectory_features(np.asarray([0]))[0, 4] == 1
    publish(np.asarray([0]), np.asarray([4]), 1.0, 4)
    observe_accepts(
        np.asarray([0]),
        np.asarray([3.0]),
        np.asarray([0.0]),
        np.ones((1, 15), np.float32),
    )
    after = trajectory_features(np.asarray([0]))
    assert after[0, 2] == 1 and after[0, 4] == 0
    bind_requests(np.asarray([0]), np.asarray([8]))
    fresh = trajectory_features(np.asarray([0]))
    assert fresh[0, 4] == 1 and fresh[0, 2] == 0
    print("ELASTIC_PLAN_OK", pooled, q, round(lam, 4))


def set_measured_draft(rows, ms) -> None:
    """Install full-step graph times. The curve stays non-decreasing."""
    global _DRAFT_ROWS, _DRAFT_MS
    rows_a = np.asarray(rows, dtype=np.float64)
    ms_a = np.asarray(ms, dtype=np.float64)
    order = np.argsort(rows_a, kind="stable")
    rows_a, ms_a = rows_a[order], ms_a[order]
    if len(rows_a) > 1:
        keep = np.ones(len(rows_a), dtype=bool)
        keep[:-1] = rows_a[:-1] != rows_a[1:]
        rows_a, ms_a = rows_a[keep], ms_a[keep]
    for i in range(1, len(ms_a)):
        if ms_a[i] < ms_a[i - 1]:
            ms_a[i] = ms_a[i - 1]
    _DRAFT_ROWS = rows_a
    _DRAFT_MS = ms_a
    _sync_gpu_tables()


_GPU_CACHE: dict[str, tuple] = {}
_GPU_DRAFT_CAP = 20
_GPU_VCOST_CAP = 8192


def _pad_1d(arr: np.ndarray, size: int) -> np.ndarray:
    out = np.empty(size, dtype=np.float32)
    n = min(len(arr), size)
    out[:n] = arr[:n]
    if n < size:
        out[n:] = arr[n - 1] if n > 0 else 0.0
    return out


def _sync_gpu_tables() -> None:
    if not _GPU_CACHE:
        return
    import torch

    r_pad = _pad_1d(_DRAFT_ROWS, _GPU_DRAFT_CAP)
    m_pad = _pad_1d(_DRAFT_MS, _GPU_DRAFT_CAP)
    if _verify_cost is not None and len(_verify_cost):
        v_pad = _pad_1d(_verify_cost, _GPU_VCOST_CAP)
    else:
        xs = np.arange(_GPU_VCOST_CAP, dtype=np.float64)
        v_pad = (_VERIFY_FLOOR_MS + _VERIFY_PER_TOKEN_MS * xs).astype(np.float32)
    for _k, (_caps, rows_t, ms_t, vcost_t, _has_vc) in _GPU_CACHE.items():
        rows_t.copy_(torch.from_numpy(r_pad))
        ms_t.copy_(torch.from_numpy(m_pad))
        vcost_t.copy_(torch.from_numpy(v_pad))


def _gpu_tables(device):
    import torch

    key = str(device)
    hit = _GPU_CACHE.get(key)
    if hit is not None:
        return hit
    caps = torch.tensor([3.0, 4.0, 4.0, 4.0], device=device, dtype=torch.float32)
    rows_t = torch.zeros(_GPU_DRAFT_CAP, device=device, dtype=torch.float32)
    ms_t = torch.zeros(_GPU_DRAFT_CAP, device=device, dtype=torch.float32)
    vcost_t = torch.zeros(_GPU_VCOST_CAP, device=device, dtype=torch.float32)
    val = (caps, rows_t, ms_t, vcost_t, True)
    _GPU_CACHE[key] = val
    _sync_gpu_tables()
    return val


def choose_total_torch(delta, decode, p_next: int, hit, prev_d, cold):
    """GPU twin of ``choose_total``. Returns query rows ``[R]`` int32 and ``lambda``.

    Nothing here reads a GPU value back to the host. The caller copies ``q`` once.
    """
    import torch

    from vllm.v1.worker.gpu.spec_decode.crl.fused import _elastic_auction_kernel

    device = delta.device
    R = delta.shape[0]
    N = 4 * R
    caps, rows_t, ms_t, vcost_t, has_vc = _gpu_tables(device)
    option = verify_mode() != "off"
    verified = (
        torch.minimum(caps, delta / ALPHA_PASS) if option else caps.expand_as(delta)
    )
    valid = decode[:, None] & (delta > 0)
    flat_v = torch.where(valid, delta, 0.0).reshape(-1)
    flat_ver = torch.where(valid, verified, 0.0).reshape(-1)
    key = torch.where(valid.reshape(-1), flat_v, -1.0e30)
    sv, order = torch.sort(key, descending=True)
    sver = flat_ver[order]
    sval = valid.reshape(-1)[order]

    if N <= 512:
        g_int = torch.zeros(R, device=device, dtype=torch.int32)
        lam = torch.empty(1, device=device, dtype=torch.float32)
        p_next_t = torch.full((1,), float(p_next), device=device, dtype=torch.float32)
        block = 512
        _elastic_auction_kernel[(1,)](
            sv,
            sver,
            sval,
            order,
            rows_t,
            ms_t,
            vcost_t,
            g_int,
            g_int,
            lam,
            decode,
            p_next_t,
            float(_VERIFY_FLOOR_MS),
            float(_VERIFY_PER_TOKEN_MS),
            R,
            N,
            M=rows_t.numel(),
            V_LEN=vcost_t.numel(),
            HAS_VCOST=1 if has_vc else 0,
            BLOCK=block,
        )
        g = g_int.to(torch.float32)
        lam = lam[0]
    else:
        dtype = delta.dtype
        srows = torch.where(sval, 4.0, 0.0)
        zero = torch.zeros(1, device=device, dtype=dtype)
        c_y = torch.cat((zero, torch.cumsum(sv, 0)))
        c_ver = torch.cat((zero, torch.cumsum(sver, 0)))
        c_rows = torch.cat((zero, torch.cumsum(srows, 0)))

        def t_draft(q):
            idx = torch.searchsorted(rows_t, q.contiguous()).clamp(
                1, rows_t.numel() - 1
            )
            span = (rows_t[idx] - rows_t[idx - 1]).clamp(min=1e-6)
            mix = ((q - rows_t[idx - 1]) / span).clamp(0, 1)
            return ms_t[idx - 1] + (ms_t[idx] - ms_t[idx - 1]) * mix

        def t_verify(n):
            if has_vc:
                idx = n.round().long().clamp(0, vcost_t.numel() - 1)
                return vcost_t[idx]
            return _VERIFY_FLOOR_MS + _VERIFY_PER_TOKEN_MS * n.clamp(min=0)

        r = decode.to(dtype).sum()
        rate = (r + c_y) / (t_draft(c_rows) + t_verify(r + c_ver)).clamp(min=1e-6)
        lam = rate.max()
        util = c_y - lam * (t_draft(c_rows) + t_verify(float(p_next) + r + c_ver))
        g_tot = torch.argmax(util)
        pos = torch.arange(N, device=device)
        take = (pos < g_tot) & sval
        req = torch.div(order, 4, rounding_mode="floor")
        g = torch.zeros(R, device=device, dtype=dtype)
        g.scatter_add_(0, req, take.to(dtype))

    active = hit & decode
    floor = torch.where(
        active & ((prev_d == 3) | (prev_d == 7)),
        3.0,
        torch.where(active & (prev_d == 11), 4.0, 0.0),
    )
    g = torch.maximum(g, floor)
    cold_ok = cold & decode & (delta[:, 0] > 0)
    g = torch.where(cold_ok, g.clamp(min=2.0), g)
    return (g * 4).to(torch.int32), lam


def choose_total_graph(
    delta,
    decode,
    p_next_t,
    hit,
    prev_d,
    cold,
    slots,
    q_out,
    d_slot,
    uq_slot,
    lam_out,
    live_t,
    max_q_rows: int,
):
    """CUDA-graph-safe auction + cap + per-slot publish (zero CPU sync)."""
    import torch

    from vllm.v1.worker.gpu.spec_decode.crl.fused import (
        _elastic_auction_kernel,
        _elastic_cap_publish_kernel,
    )

    device = delta.device
    R = delta.shape[0]
    N = 4 * R
    caps, rows_t, ms_t, vcost_t, has_vc = _gpu_tables(device)
    option = verify_mode() != "off"
    verified = (
        torch.minimum(caps, delta / ALPHA_PASS) if option else caps.expand_as(delta)
    )
    valid = decode[:, None] & (delta > 0)
    flat_v = torch.where(valid, delta, 0.0).reshape(-1)
    flat_ver = torch.where(valid, verified, 0.0).reshape(-1)
    key = torch.where(valid.reshape(-1), flat_v, -1.0e30)
    sv, order = torch.sort(key, descending=True)
    sver = flat_ver[order]
    sval = valid.reshape(-1)[order]

    g_base = torch.zeros(R, device=device, dtype=torch.int32)
    g_uncapped = torch.zeros(R, device=device, dtype=torch.int32)
    _elastic_auction_kernel[(1,)](
        sv,
        sver,
        sval,
        order,
        rows_t,
        ms_t,
        vcost_t,
        g_base,
        g_uncapped,
        lam_out,
        decode,
        p_next_t,
        float(_VERIFY_FLOOR_MS),
        float(_VERIFY_PER_TOKEN_MS),
        R,
        N,
        M=rows_t.numel(),
        V_LEN=vcost_t.numel(),
        HAS_VCOST=1 if has_vc else 0,
        BLOCK=512,
        MAX_Q_ROWS=max_q_rows,
        HAS_GU=1,
    )
    active = hit & decode
    floor = torch.where(
        active & ((prev_d == 3) | (prev_d == 7)),
        3,
        torch.where(active & (prev_d == 11), 4, 0),
    )
    gf = torch.maximum(g_base, floor)
    gu = torch.maximum(g_uncapped, floor)
    cold_ok = cold & decode & (delta[:, 0] > 0)
    gf = torch.where(cold_ok, torch.maximum(gf, torch.full_like(gf, 2)), gf)
    gu = torch.where(cold_ok, torch.maximum(gu, torch.full_like(gu, 2)), gu)
    _elastic_cap_publish_kernel[(1,)](
        g_base,
        gf.to(torch.int32),
        gu.to(torch.int32),
        slots,
        q_out,
        d_slot,
        uq_slot,
        live_t,
        R,
        MAX_Q_ROWS=max_q_rows,
        BLOCK=128,
    )


if __name__ == "__main__":
    _self_test()
