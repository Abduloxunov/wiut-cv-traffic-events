"""Where are the labelled stopped vehicles, and which static vehicle 'tubes' exist (boxes linked by overlap over
time, ignoring tracker IDs)?"""
import json
import pickle

import cv2
import numpy as np

from common import *

VEH = {"car", "bus", "truck", "motorcycle"}
OUT = DATA / "figures" / "08_stopped_vehicle"
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def tubes(df, link_iou=0.6, max_gap=5.0):
    """Static tubes: a detection joins a tube whose anchor box overlaps it by >= link_iou and was seen within max_gap."""
    veh = df[df.cls.isin(VEH)].sort_values("frame")
    live, done = [], []
    for f, g in veh.groupby("frame"):
        t = f / FPS
        done += [x for x in live if t - x["t1"] > max_gap]
        live = [x for x in live if t - x["t1"] <= max_gap]
        for b in g[["x1", "y1", "x2", "y2"]].to_numpy():
            best = max(live, key=lambda x: iou(x["box"], b), default=None)
            if best is not None and iou(best["box"], b) >= link_iou:
                best["t1"] = t; best["n"] += 1
            else:
                live.append({"box": b, "t0": t, "t1": t, "n": 1})
    return done + live


for vid in LABELS:
    df, scene, _ = data[vid]
    gt = json.load(open(DATA / "label_bundles" / LABELS[vid], encoding="utf-8"))[vid]
    sv = [e for e in gt["events"] if e[2] == "stopped_vehicle"]
    print(vid, "labelled stopped_vehicle:", ", ".join(f"{fmt(s)}-{fmt(e)}" for s, e, _ in sv))
    tb = [x for x in tubes(df) if x["t1"] - x["t0"] >= 10]
    h, w = scene.size
    img = cv2.imread(str(DATA / "runs" / vid[:-4] / "background.jpg"))
    for x in sorted(tb, key=lambda x: x["t0"]):
        x1, y1, x2, y2 = x["box"].astype(int)
        gx, gy = int((x1 + x2) / 2), min(int(y2), h - 1)
        zone = "road" if scene.road[gy, gx] else "off-road"
        zone += ", exempt(parking/bus)" if scene.exempt_stop[gy, gx] else ""
        zone += ", junction" if scene.junction[gy, gx] else ""
        cover = x["n"] / max(1, (x["t1"] - x["t0"]) * FPS / 3)
        hit = any(x["t0"] < e and x["t1"] > s for s, e, _ in sv)
        print(f"   tube {fmt(x['t0'])}-{fmt(x['t1'])} ({x['t1']-x['t0']:5.1f}s) seen {cover:4.0%} box=({x1},{y1},{x2},{y2}) {zone}")
        col = (0, 0, 255) if x["t1"] - x["t0"] > 60 else (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 6)
        cv2.putText(img, f"{fmt(x['t0'])}-{fmt(x['t1'])}", (x1, y1 - 10), 0, 1.4, col, 4)
    cv2.imwrite(str(OUT / f"static_tubes_{vid[:-4]}.jpg"), cv2.resize(img, (1920, 1080)))
