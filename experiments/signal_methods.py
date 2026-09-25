"""Compare traffic-light state methods on hand-labelled crops of signal_5 (120 crops, 3 videos, day + dusk).

Labels: experiments/data/signal5_labels.csv (visible lamp; red+amber counted as red, dark = off).
Methods:
  hsv_fixed      naive: count bright saturated pixels per hue band (fixed S/V thresholds)
  hsv_relative   brightest saturated pixels per colour band, the lit colour must win by a margin
  position       colour-blind: which third of the head (top/middle/bottom) holds the brightest pixels
  hsv+position   production (src/signals.lamp_state): colour decides, lamp position vetoes on disagreement
  knn            1-nearest-neighbour on 12x24 HSV thumbnails, trained on the other two videos (leave-one-video-out)
Reports accuracy per video and the dangerous confusion red <-> green.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signals import BANDS, colour_state, lamp_state, position_state  # noqa: E402

DATA = Path("experiments/data")


def hsv_fixed(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s >= 90) & (v >= 110)
    counts = {c: sum(int((lit & (h >= lo) & (h <= hi)).sum()) for lo, hi in r) for c, r in BANDS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] >= 0.015 * h.size else "off"


def features(crop):
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (12, 24), interpolation=cv2.INTER_AREA).astype(float).ravel()


def main():
    crops = np.load(DATA / "signal_crops.npz")["crops"]
    meta = pd.read_csv(DATA / "signal_crops.csv", index_col="id")
    labels = pd.read_csv(DATA / "signal5_labels.csv", index_col="id").label
    df = meta.loc[labels.index].assign(label=labels)

    methods = {"hsv_fixed": hsv_fixed, "hsv_relative": colour_state, "position": position_state,
               "hsv+position": lamp_state}
    for name, fn in methods.items():
        df[name] = [fn(crops[i]) for i in df.index]
    feats = {i: features(crops[i]) for i in df.index}
    knn = {}
    for vid in df.video.unique():
        train = df[df.video != vid]
        X = np.stack([feats[i] for i in train.index])
        for i in df[df.video == vid].index:
            knn[i] = train.label.iloc[int(np.argmin(np.linalg.norm(X - feats[i], axis=1)))]
    df["knn"] = pd.Series(knn)

    rows = []
    for m in ["hsv_fixed", "hsv_relative", "position", "hsv+position", "knn"]:
        row = {"method": m, "accuracy": round(float((df[m] == df.label).mean()), 3)}
        for vid, g in df.groupby("video"):
            row[vid] = round(float((g[m] == g.label).mean()), 3)
        row["red<->green errors"] = int((((df.label == "red") & (df[m] == "green")) |
                                         ((df.label == "green") & (df[m] == "red"))).sum())
        row["lit read as off"] = int(((df.label != "off") & (df[m] == "off")).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(DATA / "signal_methods_results.csv", index=False)
    wrong = df[df["hsv+position"] != df.label][["video", "t_sec", "label", "hsv+position", "hsv_relative", "position"]]
    print("\nproduction (hsv+position) errors:\n", wrong.to_string())


if __name__ == "__main__":
    main()
