"""illegal_turn v2: a turn from the wrong lane or in a prohibited direction
(start: vehicle starts turning, end: vehicle completes the turn).

Found in the labels (experiments/illegal_turn/): every left turn from approach_1 to the bottom-left road comes from
lane 0 (the leftmost lane) -- except exactly the two labelled illegal turns, which turned left from lane 1. So the
rule here is the origin-destination one from the research (entry zone + lane -> exit zone): a vehicle that leaves
approach_1 from lane >= 1 and ends on the bottom-left road (approach_3) made a left turn from the wrong lane.
Lanes are counted with the drawn lane dividers (the solid_line zones, used here only as lane boundaries).
Event = [time it leaves the approach - PRE, time it reaches the exit road + POST].
"""
import numpy as np

from rules import VEHICLES
from scene import lookup

DEFAULTS = dict(
    FROM="approach_1",
    TO="approach_3",
    TO_NEAR=400,       # px: ending this close to the exit road counts as reaching it (it is at the frame corner)
    ALLOWED_LANES=(0,),
    PRE=1.5,           # s before leaving the approach (turning starts)
    POST=0.5,
    MIN_TRACK=10,
)


class IllegalTurn:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene
        zones = {z["name"]: z for z in scene.shapes}
        self.src = scene.mask(None, shapes=[zones[self.p["FROM"]]]) if self.p["FROM"] in zones else None
        dst = zones.get(self.p["TO"])
        self.dst = scene.mask(None, margin=self.p["TO_NEAR"], shapes=[dst]) if dst else None
        self.dividers = [z["points"].astype(float) for z in sorted(scene.of_type("solid_line"), key=lambda z: z["name"])]

    def lane(self, x, y):
        """Number of lane dividers to the left-hand side of the point (0 = leftmost lane)."""
        n = 0
        for P in self.dividers:
            a, b = P[0], P[-1]
            d = b - a
            n += int(np.sign(d[0] * (y - a[1]) - d[1] * (x - a[0])) > 0)
        return n

    def detect(self, df, explain=False):
        p = self.p
        if self.src is None or self.dst is None:
            return []
        veh = df[df.cls.isin(VEHICLES)]
        segs, why = [], []
        for tid, t in veh.groupby("track_id"):
            if len(t) < p["MIN_TRACK"]:
                continue
            t = t.sort_values("t_sec")
            in_src = lookup(self.src, t.gx, t.gy)
            in_dst = lookup(self.dst, t.gx, t.gy)
            if not in_src.any() or not in_dst.any():
                continue
            last = t[in_src].iloc[-1]
            after = t[(t.t_sec > last.t_sec) & in_dst]
            if after.empty:
                continue
            lane = self.lane(last.gx, last.gy)
            if lane in p["ALLOWED_LANES"]:
                continue
            s, e = max(0.0, last.t_sec - p["PRE"]), after.t_sec.iloc[0] + p["POST"]
            segs.append((s, e))
            why.append((s, e, tid, lane))
        segs = sorted(segs)
        out = []
        for s, e in segs:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        res = [(a, b, "left turn from lane " + ",".join(str(l) for s, e, _, l in why if s < b and e > a)) for a, b in out]
        return (res, why) if explain else res
