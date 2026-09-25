"""Rule-based event proposals from tracks + scene zones (baseline v0).

Used two ways: as pre-labels that a human accepts or fixes in tools/label_tool.html, and as the
first baseline of the detection pipeline. Output format:
  {"videos": {"C3902.MP4": {"duration": 317.8, "fps": 29.97,
                            "proposals": [[start, end, label, score, why], ...]}}}

Rules (v0, deliberately simple):
  stopped_vehicle   vehicle on the road, outside bus stop / parking, still for >= 10 s, not in a queue
  jaywalking        person on the road outside crossings / pavements / islands (with a margin) for >= 1 s,
                    and not inside a vehicle box (bus passengers, riders)
  failure_to_yield  vehicle inside a crossing while a pedestrian is on the same crossing
  congestion        >= 4 vehicles standing still inside the junction for >= 5 s (a red queue never
                    stops inside the junction)
  wrong_way         moving vehicle outside the junction heading > 135 deg against the nearest lane arrow for >= 1 s

Example:
  python src/propose_events.py --run runs/C3902 --video samples/C3902.MP4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from align import estimate_homography, median_background, warp_points

VEHICLES = {"car", "bus", "truck", "motorcycle"}
STILL = 0.12          # box heights per second below which a vehicle counts as standing still
MARGIN = 45           # px (4K) added around crossings / pavements / islands for pedestrians


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder with tracks.csv")
    p.add_argument("--video", required=True, help="the video the tracks came from (for fps, duration, alignment)")
    p.add_argument("--zones", default="scene/zones.json")
    p.add_argument("--reference", default="scene/background.jpg")
    p.add_argument("--out", default="", help="default: <run>/proposals.json")
    return p.parse_args()


# ------------------------------------------------------------------ geometry
def zones_in_video(zones, H):
    """Warp every zone from reference to video coordinates."""
    out = []
    for s in zones["shapes"]:
        out.append({**s, "points": warp_points(s["points"], H).astype(np.float32)})
    return out


def masks(shapes, types, size, margin=0):
    """Binary mask (h, w) of all polygons of the given types, optionally grown by `margin` px."""
    m = np.zeros(size, np.uint8)
    for s in shapes:
        if s["type"] in types:
            cv2.fillPoly(m, [s["points"].astype(np.int32)], 1)
    if margin:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    return m.astype(bool)


def lookup(mask, x, y):
    h, w = mask.shape
    xi, yi = np.clip(x.astype(int), 0, w - 1), np.clip(y.astype(int), 0, h - 1)
    return mask[yi, xi]


# ------------------------------------------------------------------ tracks
def load_tracks(path, fps):
    df = pd.read_csv(path)
    df["cls"] = df.groupby("track_id").cls.transform(lambda s: s.mode().iloc[0])
    df["gx"], df["gy"] = (df.x1 + df.x2) / 2, df.y2
    df["bh"] = (df.y2 - df.y1).clip(lower=8)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    # speed over a +-1 s window, in box heights per second (roughly perspective-independent)
    k = max(1, int(round(fps / (df.groupby("track_id").frame.diff().median() or 3))))
    g = df.groupby("track_id")
    dx = g.gx.shift(-k) - g.gx.shift(k)
    dy = g.gy.shift(-k) - g.gy.shift(k)
    dt = (g.frame.shift(-k) - g.frame.shift(k)) / fps
    df["speed"] = (np.hypot(dx, dy) / dt / df.bh).fillna(np.nan)
    df["vx"], df["vy"] = dx / dt, dy / dt
    return df


def runs(mask_bool, times, min_len, max_gap=0.6):
    """Contiguous True runs over a sorted time series -> [(start, end)], gaps <= max_gap bridged."""
    out, start, last = [], None, None
    for t, m in zip(times, mask_bool):
        if m:
            if start is None or t - last > max_gap:
                if start is not None and last - start >= min_len:
                    out.append((start, last))
                start = t
            last = t
    if start is not None and last - start >= min_len:
        out.append((start, last))
    return out


def merge(segments, gap=1.0):
    """Union same-class segments (the ground truth reports simultaneous events as one segment)."""
    segments = sorted(segments)
    out = []
    for s, e, why in segments:
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].add(why)
        else:
            out.append([s, e, {why}])
    return [(s, e, ", ".join(sorted(w))[:80]) for s, e, w in out]


# ------------------------------------------------------------------ rules
def stopped_vehicle(df, road, exempt, fps):
    veh = df[df.cls.isin(VEHICLES)]
    on_road = lookup(road, veh.gx.values, veh.gy.values) & ~lookup(exempt, veh.gx.values, veh.gy.values)
    still = (veh.speed < STILL) & on_road
    # queue check: how many other vehicles are standing still at the same moment
    still_per_frame = veh[still].groupby("frame").size()
    segs = []
    for tid, t in veh[still.values].groupby("track_id"):
        for s, e in runs(np.ones(len(t), bool), t.t_sec.values, min_len=10.0, max_gap=1.0):
            during = t[(t.t_sec >= s) & (t.t_sec <= e)]
            queue = still_per_frame.reindex(during.frame).fillna(0).median()
            if queue >= 4:  # many others still too: a signal queue, not a stopped vehicle
                continue
            segs.append((s, e, f"{t.cls.iloc[0]} #{tid}"))
    return segs


def jaywalking(df, road, walk_ok):
    ppl = df[df.cls == "person"].copy()
    veh = df[df.cls.isin(VEHICLES | {"bicycle"})]
    ppl["on_road"] = lookup(road, ppl.gx.values, ppl.gy.values) & ~lookup(walk_ok, ppl.gx.values, ppl.gy.values)
    # drop people whose feet are inside a vehicle box in the same frame (passengers, riders)
    vb = veh.groupby("frame")[["x1", "y1", "x2", "y2"]].apply(lambda b: b.to_numpy())
    inside = []
    for f, x, y in zip(ppl.frame.values, ppl.gx.values, ppl.gy.values):
        b = vb.get(f)
        inside.append(b is not None and bool(((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3] + 10)).any()))
    ppl["in_vehicle"] = inside
    segs = []
    for tid, t in ppl.groupby("track_id"):
        for s, e in runs((t.on_road & ~t.in_vehicle).values, t.t_sec.values, min_len=1.0):
            segs.append((s, e, f"person #{tid}"))
    return segs


def failure_to_yield(df, shapes, size):
    ppl = df[df.cls == "person"]
    veh = df[df.cls.isin(VEHICLES)]
    segs = []
    for cw in (s for s in shapes if s["type"] == "crosswalk"):
        m = masks([cw], {"crosswalk"}, size)
        m_ped = masks([cw], {"crosswalk"}, size, margin=MARGIN)
        ped_frames = set(ppl.frame[lookup(m_ped, ppl.gx.values, ppl.gy.values)])
        v_in = veh[lookup(m, veh.gx.values, veh.gy.values)]
        for tid, t in v_in.groupby("track_id"):
            if t.speed.median() < STILL:  # waiting at the crossing is yielding, not failing to
                continue
            for s, e in runs(np.ones(len(t), bool), t.t_sec.values, min_len=0.3, max_gap=0.6):
                frames = t.frame[(t.t_sec >= s) & (t.t_sec <= e)]
                if any(f in ped_frames for f in frames):
                    segs.append((s, e, f"{cw['name']} {t.cls.iloc[0]} #{tid}"))
    return segs


def congestion(df, junction):
    veh = df[df.cls.isin(VEHICLES)]
    stuck = veh[lookup(junction, veh.gx.values, veh.gy.values) & (veh.speed < STILL)]
    per_t = stuck.groupby("t_sec").track_id.nunique()
    times = np.array(sorted(df.t_sec.unique()))
    flag = per_t.reindex(times).fillna(0).values >= 4
    return [(s, e, "vehicles stuck in junction") for s, e in runs(flag, times, min_len=5.0, max_gap=1.0)]


def wrong_way(df, shapes, junction):
    arrows = [s["points"] for s in shapes if s["type"] == "lane_dir"]
    if not arrows:
        return []
    veh = df[df.cls.isin(VEHICLES) & (df.speed > 0.5)].copy()
    veh = veh[~lookup(junction, veh.gx.values, veh.gy.values)]
    P = veh[["gx", "gy"]].to_numpy()
    best_d, best_dir = np.full(len(P), np.inf), np.zeros((len(P), 2))
    for a in arrows:
        p0, p1 = a[0], a[1]
        seg = p1 - p0
        t = np.clip(((P - p0) @ seg) / (seg @ seg), 0, 1)
        d = np.hypot(*(P - (p0 + t[:, None] * seg)).T)
        closer = d < best_d
        best_d[closer], best_dir[closer] = d[closer], seg / np.linalg.norm(seg)
    v = veh[["vx", "vy"]].to_numpy()
    cos = (v * best_dir).sum(1) / (np.linalg.norm(v, axis=1) + 1e-9)
    veh["against"] = (cos < -0.7) & (best_d < 250)
    segs = []
    for tid, t in veh.groupby("track_id"):
        for s, e in runs(t.against.values, t.t_sec.values, min_len=1.0):
            segs.append((s, e, f"{t.cls.iloc[0]} #{tid}"))
    return segs


def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    duration = n / fps

    run = Path(a.run)
    bg_path = run / "background.jpg"
    if not bg_path.exists():
        cv2.imwrite(str(bg_path), median_background(a.video))
    H, inliers = estimate_homography(cv2.imread(a.reference), cv2.imread(str(bg_path)))
    shapes = zones_in_video(json.loads(Path(a.zones).read_text()), H)
    print(f"{Path(a.video).name}: {duration:.1f}s @ {fps:.2f}, zones aligned with {inliers} inliers")

    df = load_tracks(run / "tracks.csv", fps)
    road = masks(shapes, {"road"}, size) & ~masks(shapes, {"island"}, size)
    junction = masks(shapes, {"intersection"}, size)
    exempt = masks(shapes, {"bus_stop", "parking"}, size)
    walk_ok = masks(shapes, {"crosswalk", "sidewalk", "island", "bus_stop"}, size, margin=MARGIN)

    found = {
        "stopped_vehicle": stopped_vehicle(df, road, exempt, fps),
        "jaywalking": jaywalking(df, road, walk_ok),
        "failure_to_yield": failure_to_yield(df, shapes, size),
        "congestion": congestion(df, junction),
        "wrong_way": wrong_way(df, shapes, junction),
    }
    proposals = []
    for label, segs in found.items():
        merged = merge(segs)
        print(f"  {label:<17} {len(segs):4d} raw -> {len(merged):3d} merged")
        proposals += [[round(s, 2), round(min(e, duration), 2), label, 0.5, why] for s, e, why in merged if e > s]
    out = Path(a.out or run / "proposals.json")
    out.write_text(json.dumps({"videos": {Path(a.video).name: {"duration": round(duration, 3), "fps": round(fps, 3),
                                                               "proposals": sorted(proposals)}}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
