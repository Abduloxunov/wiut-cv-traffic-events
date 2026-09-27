"""Render failure_to_yield v2 detections: crossing (green), vehicle (red), conflicting pedestrians (yellow)."""
import json, pickle, sys, cv2, numpy as np
from common import *
from v2.failure_to_yield import FailureToYield
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
O = DATA / "figures/09_failure_to_yield"
fmt = lambda t: f"{int(t//60)}m{t%60:04.1f}"
def render(vid, times, prefix, **kw):
    df, scene, _ = data[vid]
    res, why = FailureToYield(scene, **kw).detect(df, explain=True)
    cap = cv2.VideoCapture(str(DATA / "sample_videos" / vid))
    for t in times:
        w = [x for x in why if x[0] - 0.3 <= t <= x[1] + 0.3]
        if not w: print("none at", t); continue
        s, e, name, vtid, peds = w[0]
        f = int(round(((s + e) / 2) * FPS / 3) * 3); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
        cw = [c for c in scene.of_type("crosswalk") if c["name"] == name][0]
        cv2.polylines(im, [cw["points"].astype(np.int32)], True, (0, 255, 0), 4)
        rows = df[df.frame == f]
        for _, r in rows.iterrows():
            if r.track_id == vtid: col, th = (0, 0, 255), 6
            elif r.track_id in peds: col, th = (0, 255, 255), 5
            elif r.cls == "person": col, th = (255, 200, 0), 2
            else: continue
            cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, th)
        v = rows[rows.track_id == vtid]
        cx, cy = (int(v.gx.iloc[0]), int(v.gy.iloc[0])) if len(v) else (1920, 1080)
        x0, y0 = min(max(0, cx - 900), 3840 - 1800), min(max(0, cy - 600), 2160 - 1100)
        cv2.putText(im, f"{vid} {fmt(s)}-{fmt(e)} {name}", (x0 + 20, y0 + 60), 0, 1.6, (255, 255, 255), 4)
        cv2.imwrite(str(O / f"{prefix}_{vid[:-4]}_{fmt(s)}.jpg"), im[y0:y0 + 1100, x0:x0 + 1800]); print(prefix, vid, fmt(s), name)
if __name__ == "__main__":
    kw = dict(STEP=10, EXPAND=0.0, PET=2.0, PED_MOVING=0.3, MOVING=0.4, ON_FRAC=0.15)
    render("C3902.MP4", [53.5, 286.0], "fp_cw3", **kw)
    render("C3905.MP4", [15.0, 110.0], "fp_cw1", **kw)
