"""Build the website's EDA and results data from our saved tracks, signal timelines and dev labels.

  <site>/data/eda.json          per video: resolution, fps, duration, brightness, counts per class over time,
                                light states over time
  <site>/data/dev_labels.json   our labels of the sample videos (for the prediction-vs-label rows on the timeline)
  <site>/img/heat_<video>.jpg   motion heat map (vehicles blue, people orange) over the median background
  <site>/img/traj_<video>.jpg   vehicle trajectories coloured by direction of travel, with the drawn zones

  python tools/make_site_data.py --site demo/site
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rules import add_motion  # noqa: E402
from scene import Scene  # noqa: E402

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
LABEL_FILES = {"C3897": "labels_C3897 v2 merged.json", "C3902": "labels_C3902 done.json",
               "C3905": "labels_C3905 done.json"}


def heatmap(df, bg, sigma=25):
    h, w = bg.shape[:2]
    out = (bg * 0.45).astype(np.float32)
    for sel, colour in [(df.cls.isin(VEHICLES), np.array([255, 140, 40])), (df.cls == "person", np.array([40, 160, 255]))]:
        pts = df[sel]
        acc = np.zeros((h // 4, w // 4), np.float32)
        xs = np.clip((pts.gx / 4).astype(int), 0, w // 4 - 1)
        ys = np.clip((pts.gy / 4).astype(int), 0, h // 4 - 1)
        np.add.at(acc, (ys, xs), 1)
        acc = np.sqrt(cv2.GaussianBlur(acc, (0, 0), sigma / 2))   # sqrt: busy lanes do not drown the rest
        acc = cv2.resize(acc / (np.percentile(acc[acc > 0], 99.5) if (acc > 0).any() else 1), (w, h))
        a = np.clip(acc, 0, 1)[..., None]
        out = out * (1 - 0.8 * a) + colour * 0.8 * a
    return np.clip(out, 0, 255).astype(np.uint8)


def trajectories(df, bg, scene, max_tracks=900):
    out = (bg * 0.5).astype(np.uint8)
    for z in scene.shapes:
        if z["type"] in ("crosswalk", "stop_line", "solid_line", "island"):
            cv2.polylines(out, [z["points"].astype(np.int32)], z["type"] != "solid_line" and z["type"] != "stop_line",
                          (200, 200, 200), 3)
    veh = df[df.cls.isin(VEHICLES)]
    ids = veh.track_id.value_counts()
    ids = ids[ids >= 15].index[:max_tracks]
    for tid in ids:
        t = veh[veh.track_id == tid].sort_values("t_sec")
        p = t[["gx", "gy"]].to_numpy()
        d = p[-1] - p[0]
        if np.hypot(*d) < 150:
            continue
        hue = int((np.degrees(np.arctan2(d[1], d[0])) % 360) / 2)
        col = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(out, [p.astype(np.int32)], False, col, 3, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(ROOT / "demo" / "site"))
    ap.add_argument("--videos", nargs="+", default=["C3897", "C3902", "C3905", "C3896"])
    ap.add_argument("--runs", default=str(ROOT.parent / "runs"))
    ap.add_argument("--labels", default=str(ROOT.parent / "label_bundles"))
    a = ap.parse_args()
    site, runs = Path(a.site), Path(a.runs)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "img").mkdir(parents=True, exist_ok=True)
    eda = {}
    for v in a.videos:
        fps = 29.97
        df = add_motion(pd.read_csv(runs / v / "tracks.csv"), fps)
        bg = cv2.imread(str(runs / v / "background.jpg"))
        scene = Scene.from_image(bg)
        dur = float(df.t_sec.max())
        df["bin"] = (df.t_sec // 5 * 5).astype(int)
        per_frame = df.groupby(["bin", "frame", "cls"]).size().groupby(["bin", "cls"]).mean().unstack(fill_value=0)
        counts = {c: [round(float(x), 2) for x in per_frame[c]] for c in per_frame.columns}
        tracks = df.groupby("cls").track_id.nunique().to_dict()
        sig = pd.read_csv(runs / "signals" / f"{v}.csv") if (runs / "signals" / f"{v}.csv").exists() else None
        lights = sig[["t_sec", "signal_5"]].values.tolist() if sig is not None and "signal_5" in sig else []
        eda[v] = {"resolution": "3840x2160", "fps": fps, "duration": round(dur, 1),
                  "brightness": round(float(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).mean()), 1),
                  "bins": [int(b) for b in per_frame.index], "counts": counts,
                  "tracks": {k: int(x) for k, x in tracks.items()},
                  "lights": [[round(float(t), 1), s] for t, s in lights[::2]]}
        small = lambda im: cv2.resize(im, (1600, 900), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(site / "img" / f"heat_{v}.jpg"), small(heatmap(df, bg)), [cv2.IMWRITE_JPEG_QUALITY, 82])
        cv2.imwrite(str(site / "img" / f"traj_{v}.jpg"), small(trajectories(df, bg, scene)), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(v, "done")
    (site / "data" / "eda.json").write_text(json.dumps(eda))
    labels = {}
    for v, f in LABEL_FILES.items():
        p = Path(a.labels) / f
        if p.exists():
            labels[v] = json.load(open(p, encoding="utf-8"))[f"{v}.MP4"]["events"]
    (site / "data" / "dev_labels.json").write_text(json.dumps(labels))


if __name__ == "__main__":
    main()
