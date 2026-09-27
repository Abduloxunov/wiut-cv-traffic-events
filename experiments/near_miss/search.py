"""near_miss v2 on the dev labels (2 labelled events: look at recall and false alarms, not fine scores)."""
import itertools, json, pickle
import numpy as np
from common import *
from near_miss_prototype import NearMiss, replay
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "near_miss"; V = list(LABELS)
fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"
f1 = lambda pred, vids: score({v: pred[v] for v in vids}, {v: gts[v] for v in vids}, L)["per_class"].get(L, {"f1_mean": 0})["f1_mean"]
try:
    traces = pickle.load(open(DATA / "runs/jaywalking_v2/risk_traces.pkl", "rb"))
except FileNotFoundError:
    traces = {v: replay(data[v][0], data[v][1]) for v in V}
    pickle.dump(traces, open(DATA / "runs/jaywalking_v2/risk_traces.pkl", "wb"))
for v in V:
    tr = traces[v]; g = [e[:2] for e in gts[v]["events"] if e[2] == L]
    d = np.array([x[1] for x in tr]); t = np.array([x[0] for x in tr])
    print(v, "labels", [f"{fmt(s)}-{fmt(e)}" for s, e in g], "| DRAC 99/99.9 pct %.1f/%.1f max %.1f" % (np.percentile(d, 99), np.percentile(d, 99.9), d.max()))
    for s, e in g:
        m = (t >= s - 2) & (t <= e + 2); print(f"   around label: max DRAC {d[m].max():.1f} at {fmt(t[m][np.argmax(d[m])])}")
grid = dict(DRAC_MIN=[3, 5, 10, 20, 40], MIN_DUR=[0.0, 0.3, 0.6], EVASIVE=[False, True], DECEL=[0.5, 1.0], PRE=[0.5, 1.0], POST=[0.5, 1.5], GAP=[1.0, 3.0])
res = []
for c in itertools.product(*grid.values()):
    k = dict(zip(grid, c))
    if not k["EVASIVE"] and k["DECEL"] != 0.5: continue
    pred = {v: [(s, e) for s, e, _ in NearMiss(data[v][1], **k).detect(data[v][0], traces[v])] for v in V}
    res.append((k, pred))
for k, pred in sorted(res, key=lambda r: -f1(r[1], V))[:10]:
    print(round(f1(pred, V), 3), sum(map(len, pred.values())), "events", k)
