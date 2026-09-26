"""Frame-level diagnosis of the jaywalking signal: how often is a candidate present inside / outside labelled events,
and where do the false candidates stand (heat map over the background)."""
import numpy as np
import cv2
from common import *
from rules import _inside_vehicle, VEHICLES
from scene import lookup

OUT = DATA / "runs" / "jaywalking_v2"; OUT.mkdir(parents=True, exist_ok=True)
for vid in LABELS:
    df, scene, gt = load(vid)
    ppl = df[df.cls == "person"].copy()
    ppl["road"] = lookup(scene.road, ppl.gx, ppl.gy)
    ppl["walk_ok"] = lookup(scene.walk_ok, ppl.gx, ppl.gy)
    ppl["in_veh"] = _inside_vehicle(ppl, df[df.cls.isin(VEHICLES | {"bicycle"})])
    ppl["cand"] = ppl.road & ~ppl.walk_ok & ~ppl.in_veh
    frames = np.sort(df.frame.unique()); t = frames / FPS
    ev_ = [e for e in gt["events"] if e[2] == "jaywalking"]
    inside = np.zeros(len(t), bool)
    for s, e, _ in ev_: inside |= (t >= s) & (t <= e)
    per_f = ppl[ppl.cand].groupby("frame").size().reindex(frames, fill_value=0).to_numpy()
    print(f"{vid}: frames {len(t)}, labelled jaywalking {inside.mean():.0%} of time")
    print(f"   candidate present: inside events {np.mean(per_f[inside] > 0):.0%} of frames, outside {np.mean(per_f[~inside] > 0):.0%}")
    for s, e, _ in ev_:
        m = (t >= s) & (t <= e)
        print(f"   event {s:6.1f}-{e:6.1f} ({e-s:5.1f}s): candidate in {np.mean(per_f[m] > 0):4.0%} of frames")
    # where do outside-event candidates stand
    fr_out = set(frames[~inside])
    fp = ppl[ppl.cand & ppl.frame.isin(fr_out)]
    img = cv2.imread(str(DATA / "runs" / vid[:-4] / "background.jpg"))
    over = img.copy()
    over[scene.road & ~scene.walk_ok] = (0.6 * over[scene.road & ~scene.walk_ok] + 0.4 * np.array([0, 90, 0])).astype(np.uint8)
    for x, y in fp[["gx", "gy"]].to_numpy()[::2]:
        cv2.circle(over, (int(x), int(y)), 6, (0, 0, 255), -1)
    tp = ppl[ppl.cand & ~ppl.frame.isin(fr_out)]
    for x, y in tp[["gx", "gy"]].to_numpy()[::2]:
        cv2.circle(over, (int(x), int(y)), 6, (255, 200, 0), -1)
    cv2.imwrite(str(OUT / f"diag_{vid[:-4]}.jpg"), cv2.resize(over, (1920, 1080)))
    fp_tracks = fp.groupby("track_id").agg(n=("frame", "size"), t0=("t_sec", "min"), t1=("t_sec", "max"), x=("gx", "median"), y=("gy", "median"), bh=("bh", "median"))
    print("   top false-candidate tracks:\n" + fp_tracks.sort_values("n", ascending=False).head(8).round(0).to_string())
