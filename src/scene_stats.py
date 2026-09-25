"""Learn the scene layout from tracks and render EDA images.

Reads tracks.csv from detect_track.py and writes, next to it:
  road_heatmap.jpg     where vehicles drive (ground points), over the background
  person_heatmap.jpg   where people walk, with crosswalks outlined
  flow.jpg             mean vehicle motion per grid cell = learned lane directions
  counts.png           objects per class over time
  flow.npz             the flow field and vehicle-occupancy grid, reused by the rules

Example:
  python src/scene_stats.py --run runs/C3905
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VEHICLES = ["car", "bus", "truck", "motorcycle", "bicycle"]
REF_W, REF_H = 1280, 720


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder with tracks.csv")
    p.add_argument("--background", default="scene/background.jpg")
    p.add_argument("--zones", default="scene/zones.json", help="from tools/zone_editor.html")
    p.add_argument("--src-width", type=int, default=3840, help="width of the video the tracks came from")
    p.add_argument("--cell", type=int, default=32, help="flow grid cell size in 1280-wide pixels")
    return p.parse_args()


def ground_points(df, scale):
    """Bottom-centre of each box in 1280-wide coordinates: where the object touches the road."""
    df = df.copy()
    df["gx"] = (df.x1 + df.x2) / 2 * scale
    df["gy"] = df.y2 * scale
    return df


def heatmap_overlay(bg, xs, ys, color_map=cv2.COLORMAP_INFERNO, blur=9):
    h = np.zeros(bg.shape[:2], np.float32)
    xs = np.clip(xs.astype(int), 0, REF_W - 1)
    ys = np.clip(ys.astype(int), 0, REF_H - 1)
    np.add.at(h, (ys, xs), 1)
    h = cv2.GaussianBlur(h, (0, 0), blur)
    h = np.log1p(h)
    h = np.clip(h / max(np.percentile(h[h > 0], 99), 1e-6), 0, 1)  # percentile, so static crowds don't wash out the rest
    colored = cv2.applyColorMap((h * 255).astype(np.uint8), color_map)
    alpha = (h[..., None] * 0.85).clip(0, 0.85)
    return (bg * (1 - alpha) + colored * alpha).astype(np.uint8)


def flow_field(veh, cell):
    """Average per-step displacement of vehicle ground points in each grid cell."""
    veh = veh.sort_values(["track_id", "frame"])
    veh["dx"] = veh.groupby("track_id").gx.diff()
    veh["dy"] = veh.groupby("track_id").gy.diff()
    steps = veh.dropna(subset=["dx"])
    steps = steps[np.hypot(steps.dx, steps.dy) > 0.5]  # ignore stationary jitter
    gh, gw = REF_H // cell, REF_W // cell
    sx, sy, n = np.zeros((gh, gw)), np.zeros((gh, gw)), np.zeros((gh, gw))
    cx = np.clip((steps.gx // cell).astype(int), 0, gw - 1)
    cy = np.clip((steps.gy // cell).astype(int), 0, gh - 1)
    np.add.at(sx, (cy, cx), steps.dx)
    np.add.at(sy, (cy, cx), steps.dy)
    np.add.at(n, (cy, cx), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sx / n, sy / n, n


def draw_flow(bg, fx, fy, n, cell, min_count=5):
    img = bg.copy()
    for r in range(fx.shape[0]):
        for c in range(fx.shape[1]):
            if n[r, c] < min_count:
                continue
            v = np.array([fx[r, c], fy[r, c]])
            if np.linalg.norm(v) < 1e-3:
                continue
            v = v / np.linalg.norm(v) * cell * 0.45
            p0 = (int((c + 0.5) * cell - v[0]), int((r + 0.5) * cell - v[1]))
            p1 = (int((c + 0.5) * cell + v[0]), int((r + 0.5) * cell + v[1]))
            hue = int((np.degrees(np.arctan2(v[1], v[0])) % 360) / 2)  # colour = direction
            col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            cv2.arrowedLine(img, p0, p1, col, 2, cv2.LINE_AA, tipLength=0.4)
    return img


def draw_crosswalks(img, zones):
    if zones is None:
        return img
    sx, sy = REF_W / zones["image_size"][0], REF_H / zones["image_size"][1]
    for shape in zones["shapes"]:
        if shape["type"] == "crosswalk":
            pts = np.array([[x * sx, y * sy] for x, y in shape["points"]], np.int32)
            cv2.polylines(img, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    a = parse_args()
    run = Path(a.run)
    df = ground_points(pd.read_csv(run / "tracks.csv"), REF_W / a.src_width)
    bg = cv2.resize(cv2.imread(a.background), (REF_W, REF_H), interpolation=cv2.INTER_AREA)
    zones = json.loads(Path(a.zones).read_text()) if Path(a.zones).exists() else None

    veh = df[df.cls.isin(VEHICLES)]
    ppl = df[df.cls == "person"]
    cv2.imwrite(str(run / "road_heatmap.jpg"), heatmap_overlay(bg, veh.gx.values, veh.gy.values))
    cv2.imwrite(str(run / "person_heatmap.jpg"),
                draw_crosswalks(heatmap_overlay(bg, ppl.gx.values, ppl.gy.values, cv2.COLORMAP_OCEAN), zones))

    fx, fy, n = flow_field(veh[veh.cls != "bicycle"], a.cell)  # cyclists ride on crossings, not lanes
    cv2.imwrite(str(run / "flow.jpg"), draw_flow(bg, fx, fy, n, a.cell))
    np.savez(run / "flow.npz", fx=fx, fy=fy, n=n, cell=a.cell)

    per_sec = df.assign(sec=df.t_sec.astype(int)).groupby(["sec", "cls", "frame"]).size()
    per_sec = per_sec.groupby(["sec", "cls"]).mean().unstack(fill_value=0)  # mean objects per frame, per second
    ax = per_sec.plot(figsize=(10, 3.5), lw=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("objects in frame")
    ax.set_title(f"{run.name}: objects per frame by class")
    plt.tight_layout()
    plt.savefig(run / "counts.png", dpi=120)

    tracks = df.groupby("track_id").agg(cls=("cls", "first"), n=("frame", "size"),
                                        t0=("t_sec", "min"), t1=("t_sec", "max"))
    print(tracks.groupby("cls").agg(tracks=("n", "size"), median_obs=("n", "median"),
                                    median_life_s=("t1", lambda s: float(np.median(s - tracks.loc[s.index, "t0"])))))
    print(f"wrote EDA images to {run}")


if __name__ == "__main__":
    main()
