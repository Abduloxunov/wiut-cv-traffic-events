"""Grid search for failure_to_yield v2 with leave-one-video-out (11 labelled events: read the LOVO number)."""
import itertools, json, pickle
from common import *
from v2.failure_to_yield import FailureToYield
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "failure_to_yield"
ppl = {v: FailureToYield(data[v][1]).pedestrians(data[v][0]) for v in LABELS}
f1 = lambda pred, vids: score({v: pred[v] for v in vids}, {v: gts[v] for v in vids}, L)["per_class"].get(L, {"f1_mean": 0})["f1_mean"]
grid = dict(MODE=["pet"], STEP=[0, 10], EXPAND=[0.0, 0.15], PET=[1.0, 2.0, 3.0], PED_MOVING=[0.0, 0.3], MOVING=[0.4, 1.0], ON_FRAC=[0.15, 0.4])
res = []
for c in itertools.product(*grid.values()):
    kw = dict(zip(grid, c))
    pred = {v: [(s, e) for s, e, _ in FailureToYield(data[v][1], **kw).detect(data[v][0], ppl=ppl[v])] for v in LABELS}
    res.append((kw, pred, sum(len(x) for x in pred.values())))
V = list(LABELS)
best = max(res, key=lambda r: f1(r[1], V))
print("best on all 3:", best[0], round(f1(best[1], V), 3), "predictions", best[2])
lovo = {}
for h in V:
    tr = [v for v in V if v != h]
    b = max(res, key=lambda r: f1(r[1], tr))
    lovo[h] = b[1][h]; print(f"  without {h}: {b[0]} -> {h} {f1(b[1], [h]):.3f}")
print("LOVO pooled:", round(f1(lovo, V), 3))
top = sorted(res, key=lambda r: -f1(r[1], V))[:8]
for kw, pred, n in top: print(round(f1(pred, V), 3), n, kw)
pickle.dump(res, open(DATA / "runs/jaywalking_v2/fty_search.pkl", "wb"))
