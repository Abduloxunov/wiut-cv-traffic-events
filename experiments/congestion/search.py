"""Grid search for congestion v2 with leave-one-video-out; also scores the old rule."""
import itertools, json, pickle
from common import *
from v2.congestion import Congestion
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "congestion"; V = list(LABELS)
f1 = lambda pred, vids: score({v: pred[v] for v in vids}, {v: gts[v] for v in vids}, L)["per_class"].get(L, {"f1_mean": 0})["f1_mean"]
old = {}
for v in V:
    pr = json.load(open(f"C:/Users/Lenovo/AppData/Local/Temp/claude/D--hakathon-wiut/7454e6ba-4e25-4cf0-be36-be5d5b30d99e/scratchpad/analysis/prop_{v[:-4]}.json"))
    old[v] = [(s, e) for s, e, l, *_ in pr["videos"][v]["proposals"] if l == L]
print("old rule:", round(f1(old, V), 3))
flag_grid = dict(AREAS=[("approach",), ("approach", "intersection")], N_MIN=[3, 5, 8], STILL=[0.15, 0.3], STILL_SHARE=[0.5, 0.7], CRAWL=[0.3, 0.6, 1.0])
seg_grid = dict(WINDOW=[2.0, 4.0], RATIO=[0.5, 0.7], GAP=[2.0, 5.0, 10.0], MIN_LEN=[3.0, 8.0])
res = []
for fc in itertools.product(*flag_grid.values()):
    fk = dict(zip(flag_grid, fc))
    flags = {v: Congestion(data[v][1], **fk).frame_flags(data[v][0]) for v in V}
    for sc in itertools.product(*seg_grid.values()):
        kw = {**fk, **dict(zip(seg_grid, sc))}
        pred = {v: [(s, e) for s, e, _ in Congestion(data[v][1], **kw).detect(data[v][0], flags[v])] for v in V}
        res.append((kw, pred))
best = max(res, key=lambda r: f1(r[1], V))
print("best on all 3:", best[0], round(f1(best[1], V), 3), sum(len(x) for x in best[1].values()), "events")
lovo = {}
for h in V:
    tr = [v for v in V if v != h]; b = max(res, key=lambda r: f1(r[1], tr)); lovo[h] = b[1][h]
    print(f"  without {h}: {b[0]} -> {f1(b[1], [h]):.3f}")
print("LOVO pooled:", round(f1(lovo, V), 3))
for kw, pred in sorted(res, key=lambda r: -f1(r[1], V))[:6]: print(round(f1(pred, V), 3), kw)
