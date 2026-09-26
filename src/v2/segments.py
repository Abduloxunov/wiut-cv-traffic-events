"""Turning per-sample evidence into event segments (shared by all v2 classes).

Practice from the research (AI City winners, tIoU arithmetic): confirm evidence over a short window instead of single
frames, bridge short dropouts, merge same-class events that touch, drop blips. One split event scores at most 0.5 tIoU
per half, so bridging gaps matters more than anything else at tIoU 0.5/0.7.
"""
import numpy as np


def confirm(flags, times, window, ratio):
    """k-of-n smoothing: True where at least `ratio` of the samples in the centred `window` (seconds) are True."""
    flags = np.asarray(flags, float)
    times = np.asarray(times, float)
    if len(flags) == 0:
        return flags.astype(bool)
    csum = np.concatenate([[0.0], np.cumsum(flags)])
    lo = np.searchsorted(times, times - window / 2, "left")
    hi = np.searchsorted(times, times + window / 2, "right")
    return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1) >= ratio


def runs(flags, times, max_gap):
    """Contiguous True stretches -> [(start, end)], bridging gaps up to max_gap seconds."""
    out = []
    for t, f in zip(times, flags):
        if not f:
            continue
        if out and t - out[-1][1] <= max_gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(s, e) for s, e in out]


def union(segments, gap=0.0, min_len=0.0):
    """Merge overlapping or nearly touching segments (gap in seconds), then drop the ones shorter than min_len."""
    out = []
    for s, e in sorted(segments):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out if e - s >= min_len]
