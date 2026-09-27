"""failure_to_yield v2: a vehicle drives through a crossing while a pedestrian is on it or stepping onto it
(start: vehicle enters the crossing, end: vehicle leaves it).

Design (research: US Uniform Vehicle Code 11-502 "half of the roadway ... or so closely as to be in danger",
lane-cell yielding studies, post-encroachment gating; team decisions in labels/notes.md):
  1. vehicle on a crossing = the bottom band of its box (the part touching the road) overlaps the crossing; the
     segment runs from the first to the last such frame, so it follows the front entering and the rear leaving.
  2. only vehicles that drive through (median speed over the crossing >= MOVING); waiting before it is yielding.
  3. conflict = post-encroachment time (PET): a pedestrian on the crossing (or within STEP px: stepping onto it)
     stood at a spot that the vehicle's footprint (bottom band of its box, widened by EXPAND) covers within PET s of
     the pedestrian being there. People waiting at the kerb end are never under the footprint; a car that yielded and
     drives on after the pedestrian passed its lane has a large PET; a car cutting in front of / behind a pedestrian
     has a small one. (Mode "lane": the earlier sideways-distance test, kept for comparison.)
  4. riders (feet inside a vehicle box, or moving faster than walking: PED_MAX, e.g. a moped whose box was missed),
     standing people (< PED_MOVING), partial boxes much shorter than the vehicle (< PED_REL_H x its height: heads of
     people behind the car) and short-lived person detections (< PED_MIN samples) are ignored; events of all crossings
     are merged per class. SKIP lists crossings to ignore (experiment only).
"""
import numpy as np

from v2.common import riders
from v2.segments import union

VEHICLES = {"car", "bus", "truck", "motorcycle"}

DEFAULTS = dict(
    BAND=0.3,        # bottom share of the vehicle box that touches the road
    ON_FRAC=0.15,    # share of that band on the crossing to count as "on it"
    MOVING=0.4,      # box heights per second: slower = waiting, not driving through
    MODE="pet",      # "pet" (footprint + time gap) or "lane" (sideways distance from the path)
    PET=2.0,         # s, largest time gap between pedestrian and vehicle at the same spot
    EXPAND=0.0,     # x vehicle width added around the footprint ("so close as to be in danger")
    LANE=1.0,        # x vehicle width: sideways distance from the vehicle's path (mode "lane")
    STEP=10,         # px (4K) around the crossing: stepping onto it
    PED_MIN=3,       # samples a pedestrian must be seen in the window
    PED_MOVING=0.2,  # box heights per second: pedestrians slower than this (standing / waiting) are ignored
    PED_MAX=1.5,     # box heights per second: faster is not walking (riders whose vehicle box was missed)
    PED_REL_H=0.5,   # pedestrian box height / vehicle box height below this = partial box (0 = off)
    SKIP=("crosswalk_3",),  # never labelled there in 12.6 min; every leave-one-video-out fold skips it (review)
    GAP=0.5,         # s, merge events closer than this
    MIN_LEN=0.3,     # s
)


class FailureToYield:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene
        self.crossings = [(c["name"], scene.mask(None, shapes=[c]), scene.mask(None, margin=self.p["STEP"], shapes=[c]))
                          for c in scene.of_type("crosswalk")]


    def _on(self, rows, mask):
        """Share of each vehicle's bottom band that lies on the mask (sampled on a 5 x 3 grid)."""
        h, w = mask.shape
        fx = np.linspace(0.1, 0.9, 5)
        fy = np.linspace(1 - self.p["BAND"], 1.0, 3)
        x1, y1, x2, y2 = (rows[c].to_numpy()[:, None, None] for c in ("x1", "y1", "x2", "y2"))
        xs = np.clip(x1 + (x2 - x1) * fx[None, None, :], 0, w - 1).astype(int)
        ys = np.clip(y1 + (y2 - y1) * fy[None, :, None], 0, h - 1).astype(int)
        return mask[ys, xs].reshape(len(rows), -1).mean(axis=1)

    def pedestrians(self, df):
        """Person rows that are not riders (cacheable: independent of the parameters)."""
        ppl = df[df.cls == "person"]
        return ppl[~riders(df, ppl)]

    def detect(self, df, explain=False, ppl=None):
        p = self.p
        veh = df[df.cls.isin(VEHICLES)]
        ppl = self.pedestrians(df) if ppl is None else ppl
        if p["PED_MOVING"] > 0:
            ppl = ppl[ppl.speed.fillna(0) >= p["PED_MOVING"]]
        ppl = ppl[ppl.speed.fillna(0) <= p["PED_MAX"]]
        segs, why = [], []
        for name, cw, cw_step in self.crossings:
            if name in p["SKIP"]:
                continue
            on = self._on(veh, cw) >= p["ON_FRAC"]
            v_on = veh[on]
            h, w = cw.shape
            px = ppl.gx.clip(0, w - 1).astype(int).to_numpy()
            py = ppl.gy.clip(0, h - 1).astype(int).to_numpy()
            peds = ppl[cw_step[py, px]]
            by_frame = {f: g for f, g in peds.groupby("frame")}
            for tid, t in v_on.groupby("track_id"):
                if len(t) < 2 or t.speed.median() < p["MOVING"]:
                    continue
                pts = t[["gx", "gy"]].to_numpy(float)
                d = pts[-1] - pts[0]
                if np.hypot(*d) < 1e-6:
                    continue
                d /= np.hypot(*d)
                width = float((t.x2 - t.x1).median())
                hits = []
                if p["MODE"] == "lane":
                    for f, gx, gy in zip(t.frame.values, t.gx.values, t.gy.values):
                        g = by_frame.get(f)
                        if g is None:
                            continue
                        rel = g[["gx", "gy"]].to_numpy(float) - (gx, gy)
                        side = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])      # distance from the path line
                        hits += list(g[side < p["LANE"] * width].track_id.values)
                else:
                    t0, t1 = t.t_sec.min() - p["PET"], t.t_sec.max() + p["PET"]
                    pw = peds[(peds.t_sec >= t0) & (peds.t_sec <= t1)]
                    if len(pw):
                        P = pw[["gx", "gy", "t_sec", "bh"]].to_numpy(float)
                        e = p["EXPAND"] * width
                        for x1, y1, x2, y2, tv in t[["x1", "y1", "x2", "y2", "t_sec"]].to_numpy(float):
                            top = y2 - p["BAND"] * (y2 - y1)
                            inside = ((P[:, 0] >= x1 - e) & (P[:, 0] <= x2 + e) & (P[:, 1] >= top - e) &
                                      (P[:, 1] <= y2 + e) & (np.abs(P[:, 2] - tv) <= p["PET"]) &
                                      (P[:, 3] >= p["PED_REL_H"] * (y2 - y1)))
                            hits += list(pw.track_id.values[inside])
                if not hits:
                    continue
                counts = {k: hits.count(k) for k in set(hits)}
                ped_ok = [k for k, c in counts.items() if len(ppl[ppl.track_id == k]) >= p["PED_MIN"]]
                if ped_ok:
                    segs.append((t.t_sec.min(), t.t_sec.max()))
                    why.append((t.t_sec.min(), t.t_sec.max(), name, tid, ped_ok))
        out = union(segs, gap=p["GAP"], min_len=p["MIN_LEN"])
        res = [(a, b, "; ".join(f"{n} veh #{v} ped {sorted(pp)[:3]}" for s, e, n, v, pp in why if s < b and e > a)[:80])
               for a, b in out]
        return (res, why) if explain else res
