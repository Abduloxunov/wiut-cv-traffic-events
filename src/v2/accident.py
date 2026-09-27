"""accident v2: contact between two or more road users, or a road user and a fixed object
(start: first frame contact is visible, end: all involved objects stop moving or leave the frame).

Design (research: AI City crash detection and the open-source accident heuristics use "contact followed by a sudden
stop"; surrogate-safety work measures closing speed in object sizes): a pair of road users is a crash when
  1. they were clearly apart and closing fast (closing speed >= CLOSING sizes/s over the last second),
  2. their footprints touch (ground points closer than TOUCH x the sum of their sizes; sizes = sqrt of box area,
     so it works near and far), and
  3. both come to rest within REST_WITHIN s of the contact and stay at rest >= REST_SEC s.
A car joining a queue brakes smoothly and does not touch; vehicles passing close by do not both stop. Event = contact
-> the moment the last one comes to rest.
STATUS: NOT EMITTED. On the four sample videos (no accident in them) it still fires 3-17 times per video even with
IMPACT_SPEED 2.0 / CLOSING 1.5 -- mostly tracker ID switches that look like a sudden stop, and cars squeezing past
each other. Next: require the contact to be visible in the boxes (overlap of the vehicle footprints, not only ground
points), ignore pairs where either track starts or ends at the contact (ID switch), verify on public crash clips.
"""
import numpy as np

from rules import VEHICLES
from v2.segments import union

ROAD_USERS = VEHICLES | {"person", "bicycle"}

DEFAULTS = dict(
    TOUCH=0.35,       # contact: ground distance < TOUCH x (size_a + size_b)
    APART=1.5,        # ... after being at least APART x that distance apart within the last LOOKBACK s
    LOOKBACK=1.0,
    CLOSING=1.0,      # sizes per second of closing speed before contact
    STILL=0.15,       # box heights / s
    REST_WITHIN=1.5,  # s after contact both must be at rest (an abrupt stop)
    REST_SEC=3.0,     # s they stay at rest
    IMPACT_SPEED=1.0, # box heights / s the faster one still has in the last 0.5 s before contact (no gradual braking)
)


class Accident:
    def __init__(self, scene=None, **params):
        self.p = {**DEFAULTS, **params}

    def detect(self, df, explain=False):
        p = self.p
        d = df[df.cls.isin(ROAD_USERS)].copy()
        d["size"] = np.sqrt(((d.x2 - d.x1) * (d.y2 - d.y1)).clip(lower=16))
        by_frame = {f: g for f, g in d.groupby("frame")}
        tracks = {tid: t.sort_values("t_sec") for tid, t in d.groupby("track_id")}
        frames = sorted(by_frame)
        seen, segs, why = set(), [], []
        for f in frames:
            g = by_frame[f]
            if len(g) < 2:
                continue
            P = g[["gx", "gy"]].to_numpy(float)
            S = g["size"].to_numpy(float)
            ids = g.track_id.to_numpy()
            i, j = np.triu_indices(len(g), 1)
            dist = np.hypot(*(P[i] - P[j]).T)
            touch = p["TOUCH"] * (S[i] + S[j])
            for k in np.flatnonzero(dist < touch):
                a, b = int(ids[i[k]]), int(ids[j[k]])
                if g.cls.iloc[i[k]] == "person" and g.cls.iloc[j[k]] == "person":
                    continue  # people walking together are not a traffic accident
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                ta, tb = tracks[a], tracks[b]
                t0 = float(g.t_sec.iloc[0])
                # 1. apart and closing fast within the last second
                pa = ta[(ta.t_sec >= t0 - p["LOOKBACK"]) & (ta.t_sec <= t0)]
                pb = tb[(tb.t_sec >= t0 - p["LOOKBACK"]) & (tb.t_sec <= t0)]
                common = np.intersect1d(pa.frame.values, pb.frame.values)
                if len(common) < 3:
                    continue
                qa = pa.set_index("frame").loc[common]
                qb = pb.set_index("frame").loc[common]
                dd = np.hypot(qa.gx.values - qb.gx.values, qa.gy.values - qb.gy.values)
                scale = (qa["size"].values + qb["size"].values) / 2
                if dd[0] < p["APART"] * touch[k]:
                    continue  # already this close (queue / perspective overlap), not an impact
                closing = (dd[0] - dd[-1]) / max(qa.t_sec.values[-1] - qa.t_sec.values[0], 1e-6) / scale.mean()
                if closing < p["CLOSING"]:
                    continue
                # 2b. an impact, not a queue join: the faster one is still fast right at contact
                va = ta[(ta.t_sec > t0 - 0.5) & (ta.t_sec <= t0)].speed.fillna(0)
                vb = tb[(tb.t_sec > t0 - 0.5) & (tb.t_sec <= t0)].speed.fillna(0)
                if max(va.min() if len(va) else 0, vb.min() if len(vb) else 0) < p["IMPACT_SPEED"]:
                    continue
                # 3. both at rest soon after, and staying at rest
                rest_ok, rest_at = True, []
                for t in (ta, tb):
                    after = t[(t.t_sec > t0) & (t.t_sec <= t0 + p["REST_WITHIN"] + p["REST_SEC"])]
                    still = after.speed.fillna(0) < p["STILL"]
                    if not still.any():
                        rest_ok = False
                        break
                    first = float(after.t_sec[still].iloc[0])
                    held = after[(after.t_sec >= first) & (after.t_sec <= first + p["REST_SEC"])]
                    if first - t0 > p["REST_WITHIN"] or len(held) < 3 or (held.speed.fillna(0) >= p["STILL"]).mean() > 0.2:
                        rest_ok = False
                        break
                    rest_at.append(first)
                if not rest_ok:
                    continue
                segs.append((t0, max(rest_at) + 0.5))
                why.append((t0, max(rest_at), a, b, round(float(closing), 2)))
        res = [(s, e, "") for s, e in union(segs, gap=2.0)]
        return (res, why) if explain else res
