"""Per approach (direction) and second: vehicles present, share still, fast-vehicle speed; around the labels."""
import json, pickle
import numpy as np, pandas as pd, cv2
from common import *
from scene import lookup
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
VEH = {"car", "bus", "truck", "motorcycle"}
fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"
for v in LABELS:
    df, scene, _ = data[v]
    gt = json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v]
    lab = [e[:2] for e in gt["events"] if e[2] == "congestion"]
    veh = df[df.cls.isin(VEH)].copy(); veh["sec"] = veh.t_sec.astype(int)
    rows = []
    for ap in scene.of_type("approach"):
        m = scene.mask(None, shapes=[ap]); a = veh[lookup(m, veh.gx, veh.gy)]
        g = a.groupby("sec").agg(n=("track_id", lambda x: x.nunique() ), still=("speed", lambda s: float((s.fillna(0) < 0.15).mean())), fast=("speed", lambda s: float(np.nanpercentile(s, 80)) if s.notna().any() else 0))
        g["ap"] = ap["name"]; rows.append(g.reset_index())
    R = pd.concat(rows)
    R["label"] = R.sec.apply(lambda t: any(s <= t <= e for s, e in lab))
    print(v, "labels:", [f"{fmt(s)}-{fmt(e)}" for s, e in lab])
    print(R.groupby(["ap", "label"])[["n", "still", "fast"]].mean().round(2).to_string())
    R.to_csv(DATA / f"runs/jaywalking_v2/cong_{v[:-4]}.csv", index=False)
    # image: approaches drawn
    img = cv2.imread(str(DATA / "runs" / v[:-4] / "background.jpg"))
    for ap, col in zip(scene.of_type("approach"), [(0, 0, 255), (0, 200, 0), (255, 120, 0)]):
        cv2.polylines(img, [ap["points"].astype(np.int32)], True, col, 8); p = ap["points"].mean(0).astype(int)
        cv2.putText(img, ap["name"], tuple(p), 0, 3, col, 8)
    cv2.imwrite(str(DATA / f"figures/10_congestion/approaches_{v[:-4]}.jpg"), cv2.resize(img, (1920, 1080)))
