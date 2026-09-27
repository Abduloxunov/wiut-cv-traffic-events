"""Boundary search for every emitted class: shift the start/end of the v2 segments and merge gaps, scored with
leave-one-video-out (tIoU scoring: boundary precision dominates). Detection runs once; only the post-processing moves."""
import itertools, json, pickle
import pandas as pd
from common import *
from v2.pipeline import detect_v2
from v2.segments import union
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
V = list(LABELS)
base = {}
for v in V:
    df, scene, _ = data[v]
    sig = pd.read_csv(DATA / f"runs/signals/{v[:-4]}.csv")
    base[v] = detect_v2(df, scene, gts[v]["duration"], signals=sig)
f1 = lambda pred, vids, L: score({v: pred[v] for v in vids}, {v: gts[v] for v in vids}, L)["per_class"].get(L, {"f1_mean": 0})["f1_mean"]
grid = dict(DS=[-2, -1, -0.5, -0.3, 0, 0.3, 0.5, 1, 2], DE=[-2, -1, -0.5, 0, 0.5, 1, 2], GAP=[0, 1, 3], MIN_LEN=[0, 1, 3])
for L in sorted({e[2] for v in V for e in base[v]}):
    res = []
    for c in itertools.product(*grid.values()):
        kw = dict(zip(grid, c))
        pred = {}
        for v in V:
            segs = [(max(0.0, s + kw["DS"]), min(gts[v]["duration"], e + kw["DE"])) for s, e, l in base[v] if l == L]
            pred[v] = [(s, e) for s, e in union([x for x in segs if x[1] > x[0]], gap=kw["GAP"], min_len=kw["MIN_LEN"])]
        res.append((kw, pred))
    cur = next(r for r in res if r[0] == dict(DS=0, DE=0, GAP=0, MIN_LEN=0))
    lovo = {}
    for h in V:
        tr = [v for v in V if v != h]
        b = max(res, key=lambda r: (round(f1(r[1], tr, L), 4), -abs(r[0]["DS"]) - abs(r[0]["DE"]) - r[0]["GAP"] - r[0]["MIN_LEN"]))
        lovo[h] = b[1][h]; print(f"  {L} without {h}: {b[0]}")
    best = max(res, key=lambda r: (round(f1(r[1], V, L), 4), -abs(r[0]["DS"]) - abs(r[0]["DE"]) - r[0]["GAP"] - r[0]["MIN_LEN"]))
    print(f"{L}: current {f1(cur[1], V, L):.3f} (LOVO {f1(cur[1], V, L):.3f}) | best {f1(best[1], V, L):.3f} {best[0]} | LOVO {f1(lovo, V, L):.3f}", flush=True)
