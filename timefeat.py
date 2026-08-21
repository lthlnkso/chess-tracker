"""Per-ply clock features.

Two numbers per ply, as specified:

    0.  log1p(seconds spent on the move) -- absolute think time
    1.  fraction of remaining clock spent -- think time relative to pressure

**Alignment matters more than either feature.** Position `t` is the board
*before* move `t`, so the only clock information causally available there is how
long move `t-1` took. Feeding move `t`'s own duration as an input at position `t`
would leak the very quantity the time head is asked to predict, and the head
would score beautifully while learning nothing.

So:  input[t]  = features of move t-1   (zeros at t = 0)
     target[t] = think-time bucket of move t   -- what the head must predict

Seconds, not milliseconds: lichess clocks are whole-second only (verified, zero
decimals in 670k values), so log1p(ms) spends nearly its whole range on the
0s -> 1s step and compresses everything above 1s into a sliver. In seconds the
same values spread cleanly over 0 - 2.5.
"""

from __future__ import annotations

import numpy as np

CLK_UNKNOWN = 0xFFFF
# Two:
#   [0] log seconds spent      -- absolute pace, which is identity-bearing
#   [1] fraction of REMAINING  -- how much of what was left this move cost
#
# A third, fraction of the BASE clock, was added in bea597a to make the set
# scale-free across time controls, and is removed again now that cross-time-
# control is closed (the control arm put the ceiling at r@10 0.34 against 0.869
# same-control). Inside a single time control it carries no information: for
# 1+0 the base is always 60 s, so [2] is seconds/60 while [0] is log1p(seconds)
# -- the same scalar in a different scaling.
#
# It is not free to keep. The input width is n_planes*64 + N_TIME_FEATS, so a
# third feature makes 835 where every trained checkpoint's first layer is
# 834 -> 256, and they refuse to load rather than degrade. That includes
# ctx5_pre_elo.pt, the only checkpoint that can play at a requested rating.
N_TIME_FEATS = 2


def ms_used_per_ply(clocks: np.ndarray, tc_base_s: int, tc_inc_s: int) -> np.ndarray:
    """Milliseconds spent on each ply, from the remaining-time trace.

    lichess records time remaining *after* the move, and the increment is added
    on completion, so:  used = before - after + increment.
    A player's previous clock is two plies back; on their first move it is the
    initial time. Returns NaN where the clock is unknown or the arithmetic is
    implausible.
    """
    n = len(clocks)
    out = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or clocks[0] == CLK_UNKNOWN:
        return out

    base_cs = float(tc_base_s) * 100.0
    inc_cs = float(tc_inc_s) * 100.0
    c = clocks.astype(np.float64)

    for t in range(n):
        if clocks[t] == CLK_UNKNOWN:
            continue
        before = c[t - 2] if t >= 2 else base_cs
        used_cs = before - c[t] + inc_cs
        # Clock trace can be noisy (rounding, lag, adjournment); reject nonsense
        # rather than feed the model a negative think time.
        if used_cs < 0 or used_cs > max(base_cs, 1.0) + inc_cs:
            continue
        out[t] = used_cs * 10.0          # centiseconds -> ms
    return out


def clock_before_per_ply(clocks: np.ndarray, tc_base_s: int) -> np.ndarray:
    """Centiseconds the mover had on their clock before each ply."""
    n = len(clocks)
    out = np.full(n, np.nan, dtype=np.float32)
    base_cs = float(tc_base_s) * 100.0
    for t in range(n):
        before = clocks[t - 2] if t >= 2 else base_cs
        if t >= 2 and clocks[t - 2] == CLK_UNKNOWN:
            continue
        out[t] = float(before)
    return out


def time_features(clocks: np.ndarray, tc_base_s: int, tc_inc_s: int):
    """-> (inputs (T, 2) float32, targets (T,) float32, valid (T,) bool).

    `inputs[t]` describes move t-1; `targets[t]` describes move t.
    """
    n = len(clocks)
    ms = ms_used_per_ply(clocks, tc_base_s, tc_inc_s)
    before = clock_before_per_ply(clocks, tc_base_s)

    log_s = np.log1p(np.nan_to_num(ms, nan=0.0) / 1000.0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (ms / 10.0) / np.maximum(before, 1.0)
    frac = np.clip(np.nan_to_num(frac, nan=0.0), 0.0, 1.0).astype(np.float32)

    valid = np.isfinite(ms)

    feats = np.zeros((n, N_TIME_FEATS), dtype=np.float32)
    if n > 1:
        # shift by one ply: what the model may see at t is move t-1
        feats[1:, 0] = log_s[:-1]                # ~0-2.5, already well scaled
        feats[1:, 1] = frac[:-1]
        feats[1:][~valid[:-1]] = 0.0

    return feats, bucketize_seconds(ms / 1000.0), valid


# Log-spaced think-time buckets. Fine where bullet's mass sits (85% is 0/1/2 s),
# coarse in the tail so the same scheme still works for blitz and rapid.
TIME_EDGES = np.array([1, 2, 3, 5, 8, 13, 21, 35], dtype=np.float32)
TIME_CENTRES = np.array([0., 1., 2., 3.5, 6., 10., 16.5, 27., 45.], dtype=np.float32)
N_TIME_BINS = len(TIME_CENTRES)


def bucketize_seconds(sec: np.ndarray) -> np.ndarray:
    """Seconds -> bucket index. NaN maps to 0 and is masked out by `valid`."""
    s = np.nan_to_num(sec, nan=0.0)
    return np.digitize(s, TIME_EDGES).astype(np.int64)
