"""Render the hand-drawn scene layout and check it against real traffic from tracked sample videos.

Writes into --out:
  zones_scene.jpg   every zone drawn on the reference background
  zones_check.jpg   the same plus what the videos show: learned vehicle flow (white arrows),
                    vehicles off the drawn road (red dots), people on the road outside crossings (magenta)
and prints a verdict for each lane_dir arrow (does real traffic move that way?).

Example:
  python src/check_zones.py --zones scene/zones.json --runs runs/C3902 runs/C3905
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from align import estimate_homography, warp_points

COLORS = {"road": "#3b82f6", "crosswalk": "#22d3ee", "island": "#a3a3a3", "sidewalk": "#a78bfa",
          "intersection": "#f59e0b", "bus_stop": "#84cc16", "parking": "#64748b", "approach": "#f472b6",
          "stop_line": "#ef4444", "solid_line": "#fde047", "lane_dir": "#10b981", "signal": "#fb923c"}
FILLED = ["road", "sidewalk", "intersection", "approach", "parking", "bus_stop", "island", "crosswalk"]  # draw order
VEHICLES = {"car", "bus", "truck", "motorcycle"}
OUT_W = 1920
CELL = 96  # flow grid cell, reference pixels


def bgr(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zones", default="scene/zones.json")
    p.add_argument("--background", default="scene/background.jpg")
    p.add_argument("--runs", nargs="*", default=[], help="folders with tracks.csv from detect_track.py")
    p.add_argument("--run-backgrounds", default="scene", help="folder with bg_<video>.jpg for each run")
    p.add_argument("--out", default="runs/zones")
    return p.parse_args()


def polys(zones, *types):
    return [np.array(s["points"], np.float32) for s in zones["shapes"] if s["type"] in types]


def inside_any(points, polygons):
    hit = np.zeros(len(points), bool)
    for poly in polygons:
        hit |= np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in points])
    return hit


def draw_zones(bg, zones, scale):
    img = bg.copy()
    for t in FILLED:
        overlay = img.copy()
        for s in zones["shapes"]:
            if s["type"] == t:
                cv2.fillPoly(overlay, [(np.array(s["points"]) * scale).astype(np.int32)], bgr(COLORS[t]))
        img = cv2.addWeighted(overlay, 0.30, img, 0.70, 0)
    for s in zones["shapes"]:
        c, pts = bgr(COLORS.get(s["type"], "#ffffff")), (np.array(s["points"]) * scale).astype(np.int32)
        if s["type"] == "lane_dir":
            cv2.arrowedLine(img, tuple(pts[0]), tuple(pts[1]), c, 4, cv2.LINE_AA, tipLength=0.3)
        elif s["type"] == "signal":
            cv2.rectangle(img, tuple(pts[0]), tuple(pts[1]), c, 3)
        elif s["type"] in ("stop_line", "solid_line"):
            cv2.polylines(img, [pts], False, c, 3, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts], True, c, 2, cv2.LINE_AA)
        if s["type"] not in ("lane_dir", "solid_line"):
            x, y = pts.mean(axis=0).astype(int) if s["type"] in ("road", "approach", "intersection") else pts[0]
            cv2.putText(img, s["name"], (int(x), int(y)), 0, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s["name"], (int(x), int(y)), 0, 0.5, c, 1, cv2.LINE_AA)
    return img


def tracks_in_reference(run, ref_bg, bg_dir):
    """Load tracks.csv and move ground points (box bottom-centre) into reference coordinates."""
    df = pd.read_csv(Path(run) / "tracks.csv")
    df["cls"] = df.groupby("track_id").cls.transform(lambda s: s.mode().iloc[0])  # one class per track
    run_bg = cv2.imread(str(Path(bg_dir) / f"bg_{Path(run).name}.jpg"))
    H, n_inl = estimate_homography(ref_bg, run_bg)
    ref = warp_points(np.c_[(df.x1 + df.x2) / 2, df.y2], np.linalg.inv(H))
    df["gx"], df["gy"] = ref[:, 0], ref[:, 1]
    print(f"{Path(run).name}: {len(df)} detections, alignment inliers {n_inl}")
    return df


def flow_field(veh, shape):
    veh = veh.sort_values(["track_id", "frame"])
    d = veh.groupby("track_id")[["gx", "gy"]].diff()
    steps = veh.assign(dx=d.gx, dy=d.gy).dropna(subset=["dx"])
    steps = steps[np.hypot(steps.dx, steps.dy) > 2]  # moving, not jitter (reference px per sampled frame)
    gh, gw = shape[0] // CELL + 1, shape[1] // CELL + 1
    ok = (steps.gx >= 0) & (steps.gy >= 0) & (steps.gx < shape[1]) & (steps.gy < shape[0])
    steps = steps[ok]
    r, c = (steps.gy // CELL).astype(int), (steps.gx // CELL).astype(int)
    fx, fy, n = np.zeros((gh, gw)), np.zeros((gh, gw)), np.zeros((gh, gw))
    np.add.at(fx, (r, c), steps.dx / np.hypot(steps.dx, steps.dy))  # unit vectors: direction, not speed
    np.add.at(fy, (r, c), steps.dy / np.hypot(steps.dx, steps.dy))
    np.add.at(n, (r, c), 1)
    return fx, fy, n


def check_arrow(p0, p1, fx, fy, n):
    """Compare a drawn direction arrow with observed flow in the cells along it."""
    p0, p1 = np.array(p0), np.array(p1)
    want = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9)
    sx = sy = cnt = 0.0
    for t in np.linspace(0, 1, 12):
        x, y = p0 + t * (p1 - p0)
        r, c = int(y // CELL), int(x // CELL)
        if 0 <= r < n.shape[0] and 0 <= c < n.shape[1]:
            sx, sy, cnt = sx + fx[r, c], sy + fy[r, c], cnt + n[r, c]
    if cnt < 20:
        return "no traffic seen here", None
    got = np.array([sx, sy]) / (np.hypot(sx, sy) + 1e-9)
    angle = float(np.degrees(np.arccos(np.clip(want @ got, -1, 1))))
    coherence = float(np.hypot(sx, sy) / cnt)  # 1 = all traffic one way, ~0 = mixed directions
    if coherence < 0.4:
        verdict = "MIXED traffic directions here"
    elif angle < 45:
        verdict = "OK"
    elif angle > 135:
        verdict = "OPPOSITE to real traffic"
    else:
        verdict = "off by a large angle"
    return f"{verdict} (angle {angle:.0f} deg, {int(cnt)} steps, coherence {coherence:.2f})", got


def main():
    a = parse_args()
    zones = json.loads(Path(a.zones).read_text())
    ref_bg = cv2.imread(a.background)
    scale = OUT_W / ref_bg.shape[1]
    small_bg = cv2.resize(ref_bg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = draw_zones(small_bg, zones, scale)
    cv2.imwrite(str(out / "zones_scene.jpg"), scene, [cv2.IMWRITE_JPEG_QUALITY, 92])
    names = [s["name"] for s in zones["shapes"]]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        print("duplicate names:", dupes)
    if not a.runs:
        return

    df = pd.concat([tracks_in_reference(r, ref_bg, a.run_backgrounds) for r in a.runs], ignore_index=True)
    veh, ppl = df[df.cls.isin(VEHICLES)], df[df.cls == "person"]
    check = draw_zones((small_bg * 0.6).astype(np.uint8), zones, scale)

    fx, fy, n = flow_field(veh, ref_bg.shape)
    for r in range(n.shape[0]):
        for c in range(n.shape[1]):
            if n[r, c] < 15:
                continue
            v = np.array([fx[r, c], fy[r, c]]) / n[r, c]
            if np.linalg.norm(v) < 0.4:  # mixed directions: draw a dot
                cv2.circle(check, (int((c + .5) * CELL * scale), int((r + .5) * CELL * scale)), 3, (255, 255, 255), -1)
                continue
            ctr = np.array([(c + .5) * CELL, (r + .5) * CELL])
            v = v / np.linalg.norm(v) * CELL * 0.42
            cv2.arrowedLine(check, tuple(((ctr - v) * scale).astype(int)), tuple(((ctr + v) * scale).astype(int)),
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.4)

    road, islands = polys(zones, "road"), polys(zones, "island")
    walk_ok = polys(zones, "crosswalk", "sidewalk", "island")
    vpts = veh[["gx", "gy"]].to_numpy()[::5]
    off_road = ~inside_any(vpts, road) | inside_any(vpts, islands)
    for x, y in vpts[off_road]:
        cv2.circle(check, (int(x * scale), int(y * scale)), 2, (0, 0, 255), -1)
    ppts = ppl[["gx", "gy"]].to_numpy()[::3]
    jay = inside_any(ppts, road) & ~inside_any(ppts, walk_ok)
    for x, y in ppts[jay]:
        cv2.circle(check, (int(x * scale), int(y * scale)), 2, (255, 0, 255), -1)
    cv2.imwrite(str(out / "zones_check.jpg"), check, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nvehicle points off the drawn road (or on an island): {off_road.mean():.1%}")
    print(f"person points on the road outside crossings/sidewalks/islands: {jay.mean():.1%}")
    print("\nlane_dir arrows vs observed traffic:")
    for s in zones["shapes"]:
        if s["type"] == "lane_dir":
            verdict, _ = check_arrow(s["points"][0], s["points"][1], fx, fy, n)
            print(f"  {s['name']:<14} {verdict}")


if __name__ == "__main__":
    main()
