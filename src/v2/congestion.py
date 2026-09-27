"""congestion v2: traffic at a standstill or crawling across all lanes of a direction
(start: queue stops moving, end: queue clears).

Design (research: per-lane occupancy / speed-density, ATSPM split failure; team decisions in labels/notes.md):
  1. areas = the drawn approaches (one per direction) and optionally the junction box (gridlock);
  2. per processed frame and area: congested when at least N_MIN vehicles are in it, at least STILL_SHARE of them
     stand (speed < STILL), and even the faster ones crawl (the FAST_Q quantile of speeds < CRAWL) -- a single queued
     lane next to a flowing lane fails the last test, because the flowing lane's vehicles are fast;
  3. k-of-n confirmation over WINDOW, runs bridged over GAP, events shorter than MIN_LEN dropped, union over areas.

Result on the dev labels (experiments/congestion/search.py): this per-direction test reached 0.30 on all 3 videos but
only 0.19 leave-one-video-out, below the v1 junction rule (>= 4 still vehicles in the junction >= 5 s: 0.29). So the
default MODE is "junction": the v1 rule plus v2 gap bridging (GAP_JUNCTION = 10 s -> 0.36). C3897's congestion labels
are accepted v1 suggestions, which favours v1 timings; the per-direction mode is kept for when better labels exist.
"""
import numpy as np

from scene import lookup
from v2.segments import confirm, runs, union

VEHICLES = {"car", "bus", "truck", "motorcycle"}

DEFAULTS = dict(
    MODE="junction",       # "junction" (v1 rule + gap bridging) or "direction" (per-approach test above)
    GAP_JUNCTION=10.0,
    PAD_END=1.0,           # s the queue is still there when the last frame flagged it (boundary search, LOVO 0.36 -> 0.40)
    AREAS=("approach",),   # zone types / names used as areas; add "intersection" for gridlock in the junction
    N_MIN=5,
    STILL=0.15,            # box heights per second
    STILL_SHARE=0.6,
    CRAWL=0.5,             # box heights per second
    FAST_Q=80,
    WINDOW=4.0,
    RATIO=0.6,
    GAP=4.0,
    MIN_LEN=5.0,
)


class Congestion:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene
        self.areas = [(z["name"], scene.mask(None, shapes=[z]))
                      for z in scene.shapes if z["type"] in self.p["AREAS"] or z["name"] in self.p["AREAS"]]

    def frame_flags(self, df):
        """{area: (times, flags)} per processed frame."""
        p = self.p
        veh = df[df.cls.isin(VEHICLES)]
        frames = np.sort(df.frame.unique())
        times = df.groupby("frame").t_sec.first().reindex(frames).to_numpy()
        out = {}
        for name, m in self.areas:
            a = veh[lookup(m, veh.gx, veh.gy)]
            spd = a.speed.fillna(0)
            g = a.assign(still=spd < p["STILL"], spd=spd).groupby("frame")
            n = g.size().reindex(frames, fill_value=0).to_numpy()
            share = g.still.mean().reindex(frames, fill_value=0).to_numpy()
            fast = g.spd.quantile(p["FAST_Q"] / 100).reindex(frames, fill_value=99).to_numpy()
            out[name] = (times, (n >= p["N_MIN"]) & (share >= p["STILL_SHARE"]) & (fast < p["CRAWL"]))
        return out

    def detect(self, df, flags=None):
        p = self.p
        if p["MODE"] == "junction":
            from rules import congestion as v1_congestion
            segs = [(s, e) for s, e, _ in v1_congestion(df, self.scene)]
            return [(a, b + p["PAD_END"], "") for a, b in union(segs, gap=p["GAP_JUNCTION"])]
        flags = self.frame_flags(df) if flags is None else flags
        segs = []
        for name, (times, f) in flags.items():
            ok = confirm(f, times, p["WINDOW"], p["RATIO"])
            segs += runs(ok, times, max_gap=p["GAP"])
        return [(a, b, "") for a, b in union(segs, gap=p["GAP"], min_len=p["MIN_LEN"])]
