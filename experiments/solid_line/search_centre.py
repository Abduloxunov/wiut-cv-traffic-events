"""solid_line_crossing, centre-point variant: a lane change = the footprint centre changes side of a solid line inside
its solid stretch; event = [crossing - PRE, crossing + POST]; a vehicle whose centre stays within STRAD x its width of
the line for >= MIN_STRADDLE s counts as straddling (event = that stretch). Searched with leave-one-video-out."""
import itertools
import json
import pickle

import numpy as np

from common import *
from diagnose import position
from rules import VEHICLES
from scene import lookup
from v2.segments import runs, union

data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "solid_line_crossing"
V = list(LABELS)
f1 = lambda pred, vids: score({v: pred[v] for v in vids}, {v: gts[v] for v in vids}, L)["per_class"].get(L, {"f1_mean": 0})["f1_mean"]

# raw events once: side changes and straddles of the footprint centre, with their position along the line
raw = {}
for v in V:
    df, scene, _ = data[v]
    veh = df[df.cls.isin(VEHICLES)]
    veh = veh[~lookup(scene.junction, veh.gx, veh.gy)]
    ev = []
    for z in scene.of_type("solid_line"):
        P = z["points"].astype(float)
        for tid, t in veh.groupby("track_id"):
            if len(t) < 5:
                continue
            t = t.sort_values("t_sec")
            C = np.c_[t.gx.values, t.gy.values]
            # signed distance to the nearest piece of the polyline, and position along it
            best_d = np.full(len(C), np.inf)
            sgn = np.zeros(len(C))
            for i in range(len(P) - 1):
                a, b = P[i], P[i + 1]
                d = b - a
                u = ((C - a) @ d) / (d @ d)
                inside = (u >= 0) & (u <= 1)
                dist = (d[0] * (C[:, 1] - a[1]) - d[1] * (C[:, 0] - a[0])) / np.hypot(*d)
                better = inside & (np.abs(dist) < best_d)
                best_d[better] = np.abs(dist[better])
                sgn[better] = np.sign(dist[better])
            ok = np.isfinite(best_d)
            if ok.sum() < 3:
                continue
            w = (t.x2 - t.x1).to_numpy()
            times = t.t_sec.to_numpy()
            idx = np.flatnonzero(ok)
            s = sgn[idx]
            for k in np.flatnonzero(s[1:] != s[:-1]):
                i0, i1 = idx[k], idx[k + 1]
                tc = (times[i0] + times[i1]) / 2
                before = C[max(0, i0 - 10)]
                after = C[min(len(C) - 1, i1 + 10)]
                move = float(np.hypot(*(after - before)) / max(t.bh.median(), 1))
                ev.append(dict(kind="cross", s=tc, e=tc, pos=position(C[i1], P), move=move, name=z["name"]))
            ev_str = []
            near = ok & (best_d < 0.25 * w)
            for a0, b0 in runs(near, times, max_gap=1.0):
                m = (times >= a0) & (times <= b0)
                ev.append(dict(kind="straddle", s=a0, e=b0, pos=position(C[m].mean(0), P), move=0.0, name=z["name"]))
    raw[v] = ev
    print(v, sum(e["kind"] == "cross" for e in ev), "crossings,", sum(e["kind"] == "straddle" for e in ev), "straddles")

grid = dict(T1=[0.1, 0.15, 0.2, 0.3, 0.45], MIN_MOVE=[0.0, 0.5, 1.0], PRE=[0.5, 1.0, 2.0], POST=[0.5, 1.0, 2.0],
            MIN_STRADDLE=[3.0, 8.0, 999.0], GAP=[1.0, 3.0])
res = []
for c in itertools.product(*grid.values()):
    k = dict(zip(grid, c))
    pred = {}
    for v in V:
        segs = [(e["s"] - k["PRE"], e["e"] + k["POST"]) for e in raw[v]
                if e["kind"] == "cross" and e["pos"] <= k["T1"] and e["move"] >= k["MIN_MOVE"]]
        segs += [(e["s"], e["e"]) for e in raw[v]
                 if e["kind"] == "straddle" and e["pos"] <= k["T1"] and e["e"] - e["s"] >= k["MIN_STRADDLE"]]
        pred[v] = union([(max(0, a), b) for a, b in segs], gap=k["GAP"])
    res.append((k, pred))
best = max(res, key=lambda r: f1(r[1], V))
print("best on all 3:", best[0], round(f1(best[1], V), 3), sum(len(x) for x in best[1].values()), "events")
lovo = {}
for h in V:
    tr = [v for v in V if v != h]
    b = max(res, key=lambda r: f1(r[1], tr))
    lovo[h] = b[1][h]
    print(f"  without {h}: {b[0]} -> {f1(b[1], [h]):.3f}")
print("LOVO pooled:", round(f1(lovo, V), 3))
for k, pred in sorted(res, key=lambda r: -f1(r[1], V))[:6]:
    print(round(f1(pred, V), 3), sum(len(x) for x in pred.values()), k)
