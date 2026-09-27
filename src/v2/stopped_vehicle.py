"""stopped_vehicle v2: a vehicle stationary on the carriageway for >= 10 s, not in a queue at a signal
(start: vehicle stops, end: moves again or is removed).

Design (research: AI City stalled-vehicle winners, ATSPM queue logic):
  1. static groups ("tubes") built from vehicle boxes by overlap with the group's typical box, ignoring tracker IDs,
     which switch over minutes; a group survives short gaps (occlusion by passing traffic, missed detections).
  2. >= MIN_STILL seconds, foot point on the carriageway and outside parking / bus bays.
  3. queue test for stops shorter than MAX_QUEUE (about one signal cycle; longer is never a queue): it is a queue
     when other still vehicles stand next to it (within QUEUE_DIST vehicle widths) for at least QUEUE_SHARE of its
     stop AND drive off together with it (within QUEUE_SYNC s) -- a queue stands together and leaves together; a
     parked car has traffic moving past it, and two parked cars leave at unrelated times. Vehicles waiting
     inside the junction (to turn) do not count. Starts / ends at the video edges snap to 0 / the video end.
  4. all qualifying groups -> union into scene-level segments (the task merges simultaneous same-class events).
"""
import numpy as np

from v2.segments import union

VEHICLES = {"car", "bus", "truck", "motorcycle"}

# tuned on the 3 labelled videos (experiments/stopped_vehicle/evaluate_v2.py)
DEFAULTS = dict(
    STILL=0.3,         # box heights per second: only near-still detections can form a static group
    LINK_IOU=0.7,      # overlap with the group's typical box to join it
    MAX_GAP=10.0,      # s a group survives unseen
    MIN_STILL=10.0,    # s, task definition
    MIN_SEEN=0.5,      # share of processed frames the group must actually be seen
    QUEUE_SHARE=0.5,   # share of its stop with other still vehicles next to it = a queue
    QUEUE_DIST=3.0,    # vehicle widths
    QUEUE_SYNC=6.0,    # s, queue members drive off within this of each other
    NEIGHBOUR_MIN=3.0, # s, still groups at least this long count as neighbours
    JUNCTION=False,    # count vehicles waiting inside the junction
    EDGE=2.0,          # s, snap starts / ends this close to the video edges
    MAX_QUEUE=120.0,   # s, longer than a signal cycle is never a queue
    GAP=5.0,           # s, bridge gaps between segments
)


class StoppedVehicle:
    def __init__(self, scene, fps=29.97, **params):
        self.p = {**DEFAULTS, **params}
        self.scene, self.fps = scene, fps

    def groups(self, df):
        """Static groups: {box (typical), t0, t1, n, seen}. Only near-still detections are linked (moving traffic
        cannot form a static group, and it keeps this fast); IoU is computed against all open groups at once."""
        p = self.p
        veh = df[df.cls.isin(VEHICLES) & (df.speed.fillna(0) < p["STILL"])].sort_values("frame")
        allf = np.sort(df.frame.unique())
        step = float(np.median(np.diff(allf))) if len(allf) > 1 else 3.0
        boxes = np.zeros((0, 4)); t0 = np.zeros(0); t1 = np.zeros(0); n = np.zeros(0, int); hist = []
        done = []
        F = veh.frame.to_numpy()
        B = veh[["x1", "y1", "x2", "y2"]].to_numpy(float)
        uniq, first = np.unique(F, return_index=True)          # veh is sorted by frame
        bounds = list(first) + [len(F)]
        for k, f in enumerate(uniq):
            t = f / self.fps
            old = t - t1 > p["MAX_GAP"]
            if old.any():
                done += [dict(box=boxes[i], t0=t0[i], t1=t1[i], n=n[i]) for i in np.flatnonzero(old)]
                keep = ~old
                boxes, t0, t1, n = boxes[keep], t0[keep], t1[keep], n[keep]
                hist = [h for h, k in zip(hist, keep) if k]
            det = B[bounds[k]:bounds[k + 1]]
            if len(boxes):
                ix = np.clip(np.minimum(det[:, None, 2], boxes[None, :, 2]) - np.maximum(det[:, None, 0], boxes[None, :, 0]), 0, None)
                iy = np.clip(np.minimum(det[:, None, 3], boxes[None, :, 3]) - np.maximum(det[:, None, 1], boxes[None, :, 1]), 0, None)
                inter = ix * iy
                area_d = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                iou = inter / (area_d[:, None] + area_b[None, :] - inter + 1e-9)
            else:
                iou = np.zeros((len(det), 0))
            used = set()
            new = []
            for d in np.argsort(-iou.max(axis=1) if iou.shape[1] else np.zeros(len(det))):
                j = int(np.argmax(iou[d])) if iou.shape[1] else -1
                if j >= 0 and iou[d, j] >= p["LINK_IOU"] and j not in used:
                    used.add(j)
                    t1[j] = t; n[j] += 1; hist[j].append(det[d])
                    if len(hist[j]) % 10 == 0:              # typical box = running median (robust to jitter)
                        boxes[j] = np.median(hist[j][-50:], axis=0)
                else:
                    new.append(det[d])
            if new:
                new = np.array(new)
                boxes = np.vstack([boxes, new]); t0 = np.concatenate([t0, np.full(len(new), t)])
                t1 = np.concatenate([t1, np.full(len(new), t)]); n = np.concatenate([n, np.ones(len(new), int)])
                hist += [[b] for b in new]
        done += [dict(box=boxes[i], t0=t0[i], t1=t1[i], n=n[i]) for i in range(len(boxes))]
        for x in done:
            x["seen"] = x["n"] / max(1.0, (x["t1"] - x["t0"]) * self.fps / step + 1)
        return done

    def candidates(self, df, groups=None):
        p, s = self.p, self.scene
        h, w = s.size
        out = []
        for x in (groups if groups is not None else self.groups(df)):
            if x["t1"] - x["t0"] < p["MIN_STILL"] or x["seen"] < p["MIN_SEEN"]:
                continue
            x1, y1, x2, y2 = x["box"]
            gx, gy = int(np.clip((x1 + x2) / 2, 0, w - 1)), int(np.clip(y2, 0, h - 1))
            if not s.road[gy, gx] or s.exempt_stop[gy, gx] or (s.junction[gy, gx] and not p["JUNCTION"]):
                continue
            x["gx"], x["gy"], x["w"] = gx, gy, x2 - x1
            out.append(x)
        return out

    def is_queue(self, x, neighbours):
        """Other still vehicles next to it for at least QUEUE_SHARE of its stop."""
        p = self.p
        (x1, y1, x2, y2), w = x["box"], x["box"][2] - x["box"][0]
        cx, cy = (x1 + x2) / 2, y2
        spans = []
        for y in neighbours:
            if y is x or y["t1"] <= x["t0"] or y["t0"] >= x["t1"] or abs(y["t1"] - x["t1"]) > p["QUEUE_SYNC"]:
                continue
            b = y["box"]
            if np.hypot((b[0] + b[2]) / 2 - cx, b[3] - cy) <= p["QUEUE_DIST"] * w:
                spans.append((max(x["t0"], y["t0"]), min(x["t1"], y["t1"])))
        covered = sum(e - s for s, e in union(spans))
        return covered >= p["QUEUE_SHARE"] * (x["t1"] - x["t0"])

    def detect(self, df, duration=None):
        p = self.p
        groups = self.groups(df)
        neighbours = [g for g in groups if g["t1"] - g["t0"] >= p["NEIGHBOUR_MIN"]]
        keep = [x for x in self.candidates(df, groups)
                if x["t1"] - x["t0"] > p["MAX_QUEUE"] or not self.is_queue(x, neighbours)]
        end = duration if duration is not None else df.t_sec.max()
        spans = [(0.0 if x["t0"] <= p["EDGE"] else x["t0"], end if end - x["t1"] <= p["EDGE"] else x["t1"]) for x in keep]
        segs = union(spans, gap=p["GAP"])
        return [(a, b, f"{sum(1 for x in keep if x['t0'] < b and x['t1'] > a)} vehicle(s)") for a, b in segs]
