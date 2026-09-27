"""solid_line_crossing v2 search: raw crossings computed once on the whole drawn lines, then filtered by where along
the line they happen (the real solid stretch starts at the stop line), sideways movement and duration."""
import itertools
import json
import pickle

import numpy as np

from common import *
from diagnose import position
from v2.segments import union
from v2.solid_line import SolidLine

data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "solid_line_crossing"
V = list(LABELS)
f1 = lambda pred, vids: score({v: pred[v] for v in vids}, {v: gts[v] for v in vids}, L)["per_class"].get(L, {"f1_mean": 0})["f1_mean"]

raw = {}
for v in V:
    df, scene, _ = data[v]
    cr = SolidLine(scene, MIN_MOVE=0.0, MIN_STRADDLE=1.0, MAX_LEN=120).crossings(df)
    lines = {z["name"]: z["points"].astype(float) for z in scene.of_type("solid_line")}
    rows = []
    for s, e, name, tid in cr:
        t = df[(df.track_id == tid) & (df.t_sec >= s) & (df.t_sec <= e)]
        c0 = np.array([t.gx.iloc[0], t.gy.iloc[0]])
        c1 = np.array([t.gx.iloc[-1], t.gy.iloc[-1]])
        rows.append(dict(s=s, e=e, name=name, pos=position(c0, lines[name]),
                         move=float(np.hypot(*(c1 - c0)) / max(t.bh.median(), 1)), cls=t.cls.iloc[0]))
    raw[v] = rows
    print(v, len(rows), "raw crossings")
pickle.dump(raw, open(DATA / "runs/jaywalking_v2/solid_raw.pkl", "wb"))

grid = dict(T1=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 1.0], MIN_MOVE=[0.0, 0.3, 0.6, 1.0], MIN_DUR=[0.0, 0.5, 1.0],
            MAX_DUR=[10, 60, 120], GAP=[1.0, 3.0])
res = []
for c in itertools.product(*grid.values()):
    k = dict(zip(grid, c))
    pred = {v: union([(r["s"], r["e"]) for r in raw[v] if r["pos"] <= k["T1"] and r["move"] >= k["MIN_MOVE"]
                      and k["MIN_DUR"] <= r["e"] - r["s"] <= k["MAX_DUR"]], gap=k["GAP"]) for v in V}
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
for k, pred in sorted(res, key=lambda r: -f1(r[1], V))[:8]:
    print(round(f1(pred, V), 3), sum(len(x) for x in pred.values()), k)
