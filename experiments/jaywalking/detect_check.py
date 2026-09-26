"""Are the barely-seen labelled jaywalking events a detector-resolution problem (small people -> tiling helps)
or occlusion / zones (tiling does not help)? Runs YOLO26m three ways on frames of those events."""
import pickle
import time

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from common import *

OUT = DATA / "figures" / "07_detection_check"; OUT.mkdir(parents=True, exist_ok=True)
EVENTS = [("C3897.MP4", 31.1, 34.2), ("C3897.MP4", 65.2, 74.3), ("C3902.MP4", 140.3, 148.3),
          ("C3902.MP4", 163.0, 166.0), ("C3905.MP4", 0.0, 8.5)]
model = YOLO(str(ROOT / "weights" / "yolo26m.pt"))
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))


def detect(img, imgsz):
    r = model.predict(img, imgsz=imgsz, conf=0.25, classes=[0], verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def tiled(img, tile=1280, overlap=160):
    h, w = img.shape[:2]
    boxes, confs = [], []
    ys = list(range(0, h - tile, tile - overlap)) + [h - tile]
    xs = list(range(0, w - tile, tile - overlap)) + [w - tile]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            b, c = detect(img[y:y + tile, x:x + tile], tile)
            if len(b):
                boxes.append(b + [x, y, x, y]); confs.append(c)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    b, c = np.concatenate(boxes), np.concatenate(confs)
    keep = torchvision.ops.nms(torch.tensor(b, dtype=torch.float32), torch.tensor(c, dtype=torch.float32), 0.5).numpy()
    return b[keep], c[keep], len(xs) * len(ys)


def on_road(boxes, scene):
    walk = scene.mask({"crosswalk", "sidewalk", "island", "bus_stop"})
    dist = cv2.distanceTransform((~walk).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = scene.size
    out = []
    for x1, y1, x2, y2 in boxes:
        gx, gy = int(np.clip((x1 + x2) / 2, 0, w - 1)), int(np.clip(y2, 0, h - 1))
        out.append(bool(scene.road[gy, gx] and dist[gy, gx] > 0.25 * (y2 - y1)))
    return np.array(out, bool)


print(f"{'event':24s} {'t':>6s} | people / on-road: 1280 | 1920 | tiles")
tot = {"1280": [0, 0], "1920": [0, 0], "tiles": [0, 0]}
for vid, s, e in EVENTS:
    _, scene, _ = data[vid]
    cap = cv2.VideoCapture(str(DATA / "sample_videos" / vid))
    for t in np.linspace(s, e, 4)[1:3].round(1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS)); ok, img = cap.read()
        res = {"1280": detect(img, 1280), "1920": detect(img, 1920)}
        tb, tc, ntiles = tiled(img); res["tiles"] = (tb, tc)
        row, vis = [], img.copy()
        for name, col, th in [("tiles", (255, 255, 0), 3), ("1920", (0, 220, 255), 5), ("1280", (0, 0, 255), 8)]:
            b, _ = res[name]; r = on_road(b, scene)
            tot[name][0] += len(b); tot[name][1] += int(r.sum())
            for (x1, y1, x2, y2), rr in zip(b.astype(int), r):
                cv2.rectangle(vis, (x1 - th, y1 - th), (x2 + th, y2 + th), col, th if rr else 1)
        for name in ["1280", "1920", "tiles"]:
            b, _ = res[name]; row.append(f"{len(b):3d}/{int(on_road(b, scene).sum()):2d}")
        print(f"{vid} {s:5.1f}-{e:5.1f}  {t:6.1f} | {'  | '.join(row)}")
        cv2.putText(vis, f"{vid} t={t}s  red=1280  yellow=1920  cyan=tiles ({ntiles})  thick=on carriageway", (40, 80), 0, 2, (255, 255, 255), 5)
        cv2.imwrite(str(OUT / f"{vid[:-4]}_{t:06.1f}.jpg"), cv2.resize(vis, (2400, 1350)))
print("totals people/on-road:", tot)
t0 = time.time(); detect(img, 1280); a = time.time() - t0
t0 = time.time(); detect(img, 1920); b = time.time() - t0
t0 = time.time(); tiled(img); c = time.time() - t0
print(f"CPU time per frame: 1280 {a:.2f}s, 1920 {b:.2f}s, tiles {c:.2f}s  (ratio tiles/1280 = {c/a:.1f}x)")
