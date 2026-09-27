"""Which solid-line crossings match the labels, and where along each drawn line do they happen?"""
import json
import pickle

import cv2
import numpy as np

from common import *
from v2.solid_line import SolidLine

data = None  # loaded in main
fmt = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"


def tiou(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))


def position(c, z):
    """Fraction along polyline z of the point on it closest to c."""
    seg = np.hypot(*np.diff(z, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)] / seg.sum()
    best = None
    for i in range(len(z) - 1):
        d = z[i + 1] - z[i]
        u = float(np.clip(((c - z[i]) @ d) / (d @ d), 0, 1))
        dist = np.hypot(*(c - (z[i] + u * d)))
        if best is None or dist < best[0]:
            best = (dist, cum[i] + u * (cum[i + 1] - cum[i]))
    return best[1]




def main():
    global data
    data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
    rows = []
    for v in LABELS:
        df, scene, _ = data[v]
        gt = [e[:2] for e in json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v]["events"]
              if e[2] == "solid_line_crossing"]
        cr = SolidLine(scene).crossings(df)
        print(v, "labels", [f"{fmt(s)}-{fmt(e)}" for s, e in gt], "| raw crossings", len(cr))
        for s, e, name, tid in cr:
            t = df[(df.track_id == tid) & (df.t_sec >= s) & (df.t_sec <= e)]
            z = [q for q in scene.of_type("solid_line") if q["name"] == name][0]["points"].astype(float)
            pos = position(np.c_[t.gx.values, t.gy.values].mean(0), z)
            hit = max((tiou((s, e), g) for g in gt), default=0)
            rows.append((v, name, round(float(pos), 2), hit >= 0.3))
            if hit >= 0.3:
                print(f"   MATCH {fmt(s)}-{fmt(e)} {name} track {tid} pos {pos:.2f} tIoU {hit:.2f}")
        for g in gt:
            if not any(tiou(g, (s, e)) >= 0.3 for s, e, *_ in cr):
                print(f"   missed label {fmt(g[0])}-{fmt(g[1])}")
    for name in sorted({r[1] for r in rows}):
        m = sorted(p for _, n, p, h in rows if n == name and h)
        o = [p for _, n, p, h in rows if n == name and not h]
        print(f"{name}: matched positions {m}; unmatched {len(o)}, position 10/50/90% "
              f"{np.round(np.percentile(o, [10, 50, 90]), 2).tolist() if o else []}")
    df, scene, _ = data["C3897.MP4"]
    img = cv2.imread(str(DATA / "runs/C3897/background.jpg"))
    for z in scene.of_type("solid_line"):
        P = z["points"].astype(np.int32)
        cv2.polylines(img, [P], False, (0, 0, 255), 6)
        cv2.circle(img, tuple(int(x) for x in P[0]), 18, (0, 255, 255), -1)
        cv2.putText(img, z["name"] + " (dot = start)", tuple(int(x) for x in P[0]), 0, 2, (0, 0, 255), 5)
    cv2.imwrite(str(DATA / "figures/11_solid_line/solid_lines_drawn.jpg"), cv2.resize(img, (1920, 1080)))


if __name__ == "__main__":
    main()
