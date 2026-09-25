"""Causal collision risk from tracked road users (Part B).

At each update the estimator sees only the detections so far. For every pair of road users it extrapolates
both at constant velocity and computes the time of closest approach (TCA) and the distance at that moment;
if that distance is smaller than the two objects' sizes, TCA is a time-to-collision (TTC). Distances and speeds
are measured in object sizes (sqrt of box area), so the same thresholds work near and far in the image.
Positions are ground points (bottom-centre of the box): boxes of cars in neighbouring lanes or in a queue overlap
in the image through perspective without touching, so pairs that already look "in contact" are ignored and only
pairs that are apart now but will meet raise risk (first version without this: 89 false alarms per hour).

A pair raises risk only when it is closing fast, will actually touch, and does so soon; the frame risk is the
strongest pair, smoothed and required to persist, so a single noisy box does not ring the alarm. The score is
built to stay near 0 in normal traffic: every alarm outside the 10 s before a real crash is a false alarm.
"""
from collections import defaultdict, deque

import numpy as np

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
HISTORY_SEC = 1.2          # velocity = least-squares slope over this window
MIN_POINTS = 8             # observations (~0.8 s at 10 Hz): new or briefly seen tracks have noisy velocities
MIN_SIZE = 0.018           # x frame width: far, tiny boxes turn pixel jitter into huge "closing speeds"
MIN_CLOSING = 0.8          # sizes per second: slower approaches are normal queueing / yielding
MIN_SPEED = 1.0            # sizes per second: at least one of the pair must clearly be moving
DRAC_HALF = 20.0           # sizes/s^2 that maps to score 0.5: above the 99.97th percentile of normal sample traffic
MAX_TTC = 4.0              # s; beyond this a pair contributes nothing
TOUCH = 0.35               # objects "touch" when their distance < TOUCH * (size_i + size_j)
APART = 1.5                # a pair must be at least this many touch-distances apart now to count
EMA = 0.5                  # smoothing of the frame risk (weight of the newest value)
PERSIST = 2                # the same pair must be on a collision course this many updates in a row
STALE_SEC = 0.5            # tracks not seen for this long are ignored


class CollisionRisk:
    def __init__(self, scene=None):
        """scene (optional): zones in this video's pixels. Road users on two different one-way carriageways
        (the `approach` zones, separated by the median) cannot collide there, so such pairs are ignored
        unless one of them is inside the junction."""
        self.carriageway = None
        if scene is not None and scene.of_type("approach"):
            self.carriageway = np.zeros(scene.size, np.int16)
            for k, s in enumerate(scene.of_type("approach"), start=1):
                self.carriageway[scene.mask(None, shapes=[s])] = k
            self.junction = scene.junction
        self.hist = defaultdict(lambda: deque())
        self.min_size = MIN_SIZE * (scene.size[1] if scene is not None else 3840)
        self.cls = {}
        self.smoothed = 0.0
        self.streak = {}   # (id_a, id_b) -> consecutive updates on a collision course

    def update(self, t, rows):
        """rows: [frame, t, track_id, cls, conf, x1, y1, x2, y2]. Returns (risk, details of the riskiest pair)."""
        for _, _, tid, cls, _, x1, y1, x2, y2 in rows:
            if cls not in VEHICLES and cls != "person":
                continue
            h = self.hist[tid]
            h.append((t, (x1 + x2) / 2, y2, max(np.sqrt((x2 - x1) * (y2 - y1)), 4.0)))
            while h and t - h[0][0] > HISTORY_SEC:
                h.popleft()
            self.cls[tid] = cls
        for tid in [k for k, h in self.hist.items() if not h or t - h[-1][0] > STALE_SEC]:
            del self.hist[tid]

        pos, vel, size, ids = [], [], [], []
        for tid, h in self.hist.items():
            if len(h) < MIN_POINTS or np.mean([o[3] for o in h]) < self.min_size:
                continue
            a = np.array(h)
            tt = a[:, 0] - a[:, 0].mean()
            denom = (tt ** 2).sum()
            if denom <= 0:
                continue
            pos.append(a[-1, 1:3])
            vel.append([(tt * (a[:, 1] - a[:, 1].mean())).sum() / denom, (tt * (a[:, 2] - a[:, 2].mean())).sum() / denom])
            size.append(a[:, 3].mean())
            ids.append(tid)
        on_course = self._pairs(np.array(pos), np.array(vel), np.array(size), ids) if len(ids) >= 2 else {}
        if self.carriageway is not None and on_course:
            where = {tid: p for tid, p in zip(ids, pos)}
            on_course = {k: v for k, v in on_course.items() if not self._separated(where[k[0]], where[k[1]])}
        self.streak = {k: self.streak.get(k, 0) + 1 for k in on_course}
        raw, best = 0.0, None
        for k, (r, info) in on_course.items():
            if self.streak[k] >= PERSIST and r > raw:
                raw, best = r, {"ids": k, **info}
        self.smoothed = EMA * raw + (1 - EMA) * self.smoothed
        return float(np.clip(self.smoothed, 0.0, 1.0)), best

    def _separated(self, a, b):
        h, w = self.carriageway.shape
        ya, xa = min(max(int(a[1]), 0), h - 1), min(max(int(a[0]), 0), w - 1)
        yb, xb = min(max(int(b[1]), 0), h - 1), min(max(int(b[0]), 0), w - 1)
        ca, cb = self.carriageway[ya, xa], self.carriageway[yb, xb]
        return ca > 0 and cb > 0 and ca != cb and not self.junction[ya, xa] and not self.junction[yb, xb]

    def _pairs(self, pos, vel, size, ids):
        """{(id_a, id_b): (risk, info)} for pairs that are apart now and will touch within MAX_TTC."""
        i, j = np.triu_indices(len(ids), k=1)
        people = np.array([self.cls[k] == "person" for k in ids])
        keep = ~(people[i] & people[j])  # person-person contact is not a traffic accident
        i, j = i[keep], j[keep]
        if i.size == 0:
            return {}
        scale = (size[i] + size[j]) / 2
        p = (pos[j] - pos[i]) / scale[:, None]            # relative position, in sizes
        v = (vel[j] - vel[i]) / scale[:, None]            # relative velocity, sizes / s
        touch = TOUCH * (size[i] + size[j]) / scale
        dist = np.sqrt((p ** 2).sum(1))
        closing = -(p * v).sum(1) / (dist + 1e-9)          # sizes / s towards each other
        # time until the distance shrinks to `touch` at constant velocity: |p + v t| = touch
        a, b, c = (v ** 2).sum(1), 2 * (p * v).sum(1), dist ** 2 - touch ** 2
        disc = b ** 2 - 4 * a * c
        ttc = np.where((a > 1e-6) & (disc >= 0), (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * np.maximum(a, 1e-6)), np.inf)
        speed = np.sqrt((vel ** 2).sum(1)) / size
        risky = (np.maximum(speed[i], speed[j]) > MIN_SPEED) & (dist > APART * touch) & (closing > MIN_CLOSING)             & (ttc > 0) & (ttc < MAX_TTC)
        out = {}
        for k in np.flatnonzero(risky):
            key = tuple(sorted((ids[i[k]], ids[j[k]])))
            # DRAC: the deceleration needed to stop before contact = closing^2 / (2 * gap), a standard surrogate
            # safety measure; calm braking into a queue needs little, a real conflict needs a lot
            gap = max(float(dist[k] - touch[k]), 0.05)
            drac = float(closing[k]) ** 2 / (2 * gap)
            out[key] = (float(1 - np.exp(-np.log(2) * drac / DRAC_HALF)),
                        {"ttc": round(float(ttc[k]), 2), "closing": round(float(closing[k]), 2), "drac": round(drac, 1)})
        return out
