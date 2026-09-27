"""solid_line_crossing v2: a lane change or manoeuvre across a solid marking
(start: a wheel crosses the line, end: the vehicle is fully in the new lane).

Design (research: line-crossing via side tests against the marking polyline):
  * the vehicle's footprint centre (bottom-centre of the box) is tested against each solid line; box corners are not
    wheels from this oblique camera (a box is wider than the car), which produced hundreds of false crossings;
  * lane change = the centre changes side of the line; event = [crossing - PRE, crossing + POST] (wheel crosses ...
    fully in the new lane);
  * straddling = centre within STRAD x vehicle width of the line for >= MIN_STRADDLE s (e.g. waiting on the line at a
    red light) -> event over that stretch;
  * only the solid stretch counts: the first T1 of each drawn line from its start at the stop line (the drawn lines run
    to the frame edge, but the real marking is solid only near the stop line -- lead's note; redraw pending);
  * vehicles inside the junction are ignored.

Dev labels (experiments/solid_line/): best 0.14 on all 3 videos, 0.0 leave-one-video-out -- weak; kept because the
class occurs in every sample video (emitting it cannot lower the score if it is in the test set). Redo after the solid
lines are redrawn.
"""
import numpy as np

from rules import VEHICLES
from scene import lookup
from v2.segments import runs, union

DEFAULTS = dict(
    T1=0.2,             # solid part = first T1 of each drawn line (fraction of its length)
    PRE=1.0,            # s before the centre crosses (wheel crosses first)
    POST=0.5,           # s after (fully in the new lane)
    STRAD=0.25,         # x vehicle width: centre this close to the line = straddling
    MIN_STRADDLE=8.0,   # s
    MIN_TRACK=5,        # samples
    GAP=3.0,
)


def position(c, z):
    """Fraction along polyline z of the point on it closest to c."""
    seg = np.hypot(*np.diff(z, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)] / seg.sum()
    best = None
    for i in range(len(z) - 1):
        d = z[i + 1] - z[i]
        u = float(np.clip(((c - z[i]) @ d) / (d @ d), 0, 1))
        dist = np.hypot(*(c - (z[i] + u * d)))
        if best is None or dist < best[0]:
            best = (dist, cum[i] + u * (cum[i + 1] - cum[i]))
    return best[1]


class SolidLine:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene

    def raw(self, df):
        """[(kind, start, end, position along the line, line name)] for all side changes and straddles."""
        p = self.p
        veh = df[df.cls.isin(VEHICLES)]
        veh = veh[~lookup(self.scene.junction, veh.gx, veh.gy)]
        out = []
        for z in self.scene.of_type("solid_line"):
            P = z["points"].astype(float)
            lo, hi = P.min(0) - 600, P.max(0) + 600
            for tid, t in veh.groupby("track_id"):
                if len(t) < p["MIN_TRACK"]:
                    continue
                t = t.sort_values("t_sec")
                C = np.c_[t.gx.values, t.gy.values]
                if not ((C >= lo) & (C <= hi)).all(axis=1).any():
                    continue
                best_d = np.full(len(C), np.inf)
                sgn = np.zeros(len(C))
                for i in range(len(P) - 1):
                    a, b = P[i], P[i + 1]
                    d = b - a
                    u = ((C - a) @ d) / (d @ d)
                    dist = (d[0] * (C[:, 1] - a[1]) - d[1] * (C[:, 0] - a[0])) / np.hypot(*d)
                    better = (u >= 0) & (u <= 1) & (np.abs(dist) < best_d)
                    best_d[better] = np.abs(dist[better])
                    sgn[better] = np.sign(dist[better])
                ok = np.flatnonzero(np.isfinite(best_d))
                if len(ok) < 3:
                    continue
                times = t.t_sec.to_numpy()
                s = sgn[ok]
                for k in np.flatnonzero(s[1:] != s[:-1]):
                    i1 = ok[k + 1]
                    tc = (times[ok[k]] + times[i1]) / 2
                    out.append(("cross", tc, tc, position(C[i1], P), z["name"]))
                near = np.isfinite(best_d) & (best_d < p["STRAD"] * (t.x2 - t.x1).to_numpy())
                for a0, b0 in runs(near, times, max_gap=1.0):
                    m = (times >= a0) & (times <= b0)
                    out.append(("straddle", a0, b0, position(C[m].mean(0), P), z["name"]))
        return out

    def detect(self, df, raw=None):
        p = self.p
        raw = self.raw(df) if raw is None else raw
        segs = [(s - p["PRE"], e + p["POST"]) for k, s, e, pos, _ in raw if k == "cross" and pos <= p["T1"]]
        segs += [(s, e) for k, s, e, pos, _ in raw if k == "straddle" and pos <= p["T1"] and e - s >= p["MIN_STRADDLE"]]
        return [(a, b, "") for a, b in union([(max(0.0, a), b) for a, b in segs], gap=p["GAP"])]
