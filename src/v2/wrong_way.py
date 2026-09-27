"""wrong_way v2: a vehicle moves against the traffic direction of its lane, including driving in the oncoming lane
(start: vehicle enters the opposing lane, end: vehicle returns to a correct lane or leaves the frame).

Design (research: wrong-way systems compare a motion vector over a window with the lane direction, gate it by
minimum displacement and confirm it over several frames; direction vectors are drawn or learned per lane):
  * only on the straight carriageways (the drawn approach zones), never in the junction, where turning vehicles
    legitimately cross every direction (the first rule fired there: turns and the lorry squeeze in C3905 / C3896);
  * motion = displacement over +-WIN/2 s (not one frame), compared with the nearest drawn lane arrow;
    against the lane = cosine below COS (about 135 degrees or more);
  * confirmed: against the lane for at least MIN_SEC and at least MIN_BACK box heights of backward travel.
Checked on all four sample videos (no wrong-way driving in them): fires 0 times.
"""
import numpy as np

from rules import VEHICLES
from scene import lookup
from v2.segments import runs, union

DEFAULTS = dict(
    WIN=2.0,        # s window for the motion vector
    COS=-0.7,       # cosine with the lane direction below this = against it
    MIN_MOVE=0.5,   # box heights per second: moving, not jitter of a standing vehicle
    MIN_SEC=2.0,    # s against the lane
    MIN_BACK=3.0,   # box heights travelled against the lane
    LANE_DIST=3.0,  # x box height: nearest lane arrow must be this close
)


class WrongWay:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene
        self.arrows = [(z["points"][0], z["points"][-1]) for z in scene.of_type("lane_dir")]
        self.carriageway = scene.mask({"approach"}) & ~scene.junction

    def detect(self, df, explain=False):
        p = self.p
        if not self.arrows:
            return ([], []) if explain else []
        veh = df[df.cls.isin(VEHICLES)]
        veh = veh[lookup(self.carriageway, veh.gx, veh.gy)]
        segs, why = [], []
        for tid, t in veh.groupby("track_id"):
            if len(t) < 10:
                continue
            t = t.sort_values("t_sec")
            T = t.t_sec.to_numpy()
            P = t[["gx", "gy"]].to_numpy(float)
            bh = t.bh.to_numpy(float)
            lo = np.searchsorted(T, T - p["WIN"] / 2)
            hi = np.clip(np.searchsorted(T, T + p["WIN"] / 2, side="right") - 1, 0, len(T) - 1)
            disp = P[hi] - P[lo]
            dt = np.maximum(T[hi] - T[lo], 1e-6)
            speed = np.hypot(*disp.T) / dt / bh
            best_d = np.full(len(P), np.inf)
            lane = np.zeros((len(P), 2))
            for a, b in self.arrows:
                seg = b - a
                u = np.clip(((P - a) @ seg) / (seg @ seg), 0, 1)
                d = np.hypot(*(P - (a + u[:, None] * seg)).T)
                closer = d < best_d
                best_d[closer] = d[closer]
                lane[closer] = seg / np.linalg.norm(seg)
            cos = (disp * lane).sum(1) / (np.hypot(*disp.T) + 1e-9)
            against = (cos < p["COS"]) & (speed >= p["MIN_MOVE"]) & (best_d <= p["LANE_DIST"] * bh)
            for s, e in runs(against, T, max_gap=0.5):
                if e - s < p["MIN_SEC"]:
                    continue
                m = (T >= s) & (T <= e)
                back = -((P[m][-1] - P[m][0]) @ lane[m].mean(0)) / np.median(bh[m])
                if back >= p["MIN_BACK"]:
                    segs.append((s, e))
                    why.append((s, e, tid, t.cls.iloc[0], round(float(back), 1)))
        res = [(s, e, "") for s, e in union(segs, gap=1.0)]
        return (res, why) if explain else res
