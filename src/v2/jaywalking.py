"""jaywalking v2: a pedestrian on the carriageway outside a crossing (start: steps onto the road, end: leaves it).

Design (see docs/research_systems / reports):
  1. per detection: foot point on the carriageway, farther than MARGIN x person-height from any place a pedestrian may
     be (crossings, pavements, islands, bus stops) and from the carriageway edge (people waiting at a kerb, drawn
     pavement or not) -- the margin scales with perspective instead of fixed pixels; not a rider (feet inside a
     vehicle / bicycle box); box not cut by the image border and not much shorter than a whole person at that image
     row (occluded boxes put the "feet" mid-body).
  2. per pedestrian: k-of-n confirmation over a short window, then runs of at least PERSON_MIN seconds;
     pedestrians whose track barely moves for a long time are dropped (static false detections, ID glue).
  3. per scene: union of all pedestrians' runs = one event while anyone is jaywalking (the task merges simultaneous
     jaywalkers into one segment), gaps up to GAP bridged, blips shorter than MIN_LEN dropped.
"""
import cv2
import numpy as np

from v2.segments import confirm, runs, union

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
WALK_TYPES = {"crosswalk", "sidewalk", "island", "bus_stop"}

# tuned on the 3 labelled videos with leave-one-video-out (experiments/jaywalking/evaluate_v2.py)
DEFAULTS = dict(
    MARGIN=0.25,        # x person box height: tolerance around crossings / pavements (about 0.8 m)
    WINDOW=1.0,        # s, confirmation window
    RATIO=0.4,         # share of samples in the window that must be on the road
    PERSON_MIN=1.0,    # s, shortest on-road run of one pedestrian
    STATIC_SEC=20.0,   # s, a "pedestrian" standing within STATIC_MOVE heights for longer is ignored
    STATIC_MOVE=0.5,
    GAP=4.0,           # s, bridge gaps between runs (scene level)
    MIN_LEN=3.0,       # s, shortest event
    BORDER=4,          # px, boxes touching the image border are ignored
    PARTIAL=0.0,       # boxes shorter than this x the expected person height at their row are ignored (0 = off)
    KERB=False,        # also keep MARGIN away from the carriageway edge (not only from walk areas)
)


class Jaywalking:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene
        walk = scene.mask(WALK_TYPES)
        self.carriageway = scene.road
        # distance (px) to the nearest place a pedestrian may be (walk areas; with KERB also any non-road pixel)
        free = (scene.road & ~walk) if self.p["KERB"] else ~walk
        self.dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    def _riders(self, ppl, vehicles):
        """True where a person's feet fall inside a vehicle/bicycle box of the same frame."""
        boxes = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in vehicles.groupby("frame")}
        out = np.zeros(len(ppl), bool)
        for i, (f, x, y) in enumerate(zip(ppl.frame.values, ppl.gx.values, ppl.gy.values)):
            b = boxes.get(f)
            if b is not None:
                out[i] = bool(((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3] + 10)).any())
        return out

    def evidence(self, df):
        """Per person detection: is this a jaywalking sample? Returns the person rows with a boolean `on_road`."""
        p, (h, w) = self.p, self.scene.size
        ppl = df[df.cls == "person"].copy()
        xi = ppl.gx.clip(0, w - 1).astype(int).to_numpy()
        yi = ppl.gy.clip(0, h - 1).astype(int).to_numpy()
        on_cw = self.carriageway[yi, xi]
        far = self.dist[yi, xi] > p["MARGIN"] * ppl.bh.to_numpy()
        # perspective model of person height: median box height per band of image rows, from this video's people
        rows = (ppl.gy // 100).astype(int)
        expected = ppl.groupby(rows).bh.transform(lambda b: b.quantile(0.7)).to_numpy()
        whole = ppl.bh.to_numpy() >= p["PARTIAL"] * expected
        border = ((ppl.x1 <= p["BORDER"]) | (ppl.x2 >= w - p["BORDER"]) | (ppl.y2 >= h - p["BORDER"])).to_numpy()
        rider = self._riders(ppl, df[df.cls.isin(VEHICLE_CLASSES)])
        ppl["on_road"] = on_cw & far & whole & ~border & ~rider
        return ppl

    def person_runs(self, ppl):
        p, out = self.p, []
        for tid, t in ppl.groupby("track_id"):
            if not t.on_road.any():
                continue
            times = t.t_sec.to_numpy()
            ok = confirm(t.on_road.to_numpy(), times, p["WINDOW"], p["RATIO"])
            for s, e in runs(ok, times, max_gap=p["WINDOW"]):
                if e - s < p["PERSON_MIN"]:
                    continue
                seg = t[(t.t_sec >= s) & (t.t_sec <= e)]
                spread = np.hypot(seg.gx.max() - seg.gx.min(), seg.gy.max() - seg.gy.min()) / seg.bh.median()
                if e - s > p["STATIC_SEC"] and spread < p["STATIC_MOVE"]:
                    continue
                out.append((s, e, tid))
        return out

    def detect(self, df):
        """-> [(start, end, why)] scene-level jaywalking events."""
        p = self.p
        pr = self.person_runs(self.evidence(df))
        segs = union([(s, e) for s, e, _ in pr], gap=p["GAP"], min_len=p["MIN_LEN"])
        return [(s, e, ", ".join(f"person #{tid}" for a, b, tid in pr if a < e and b > s)[:80]) for s, e in segs]
