"""Score jaywalking v2 on the dev labels: defaults, old rule, grid search with leave-one-video-out."""
import itertools
import json
import pickle

from common import *
from v2.jaywalking import DEFAULTS, Jaywalking

CACHE = DATA / "runs" / "jaywalking_v2" / "cache.pkl"
try:
    data = pickle.load(open(CACHE, "rb"))
except FileNotFoundError:
    data = {v: load(v) for v in LABELS}
    pickle.dump(data, open(CACHE, "wb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}  # always the latest labels


def f1(res):
    return res["per_class"]["jaywalking"]["f1_mean"] if "jaywalking" in res["per_class"] else 0.0


# evidence depends only on MARGIN: compute once per value
ev_cache = {}
def run(params, vids):
    out = {}
    for v in vids:
        df, scene, _ = data[v]
        jw = Jaywalking(scene, **params)
        key = (v, params.get("MARGIN", DEFAULTS["MARGIN"]), params.get("PARTIAL", DEFAULTS["PARTIAL"]), params.get("KERB", DEFAULTS["KERB"]))
        if key not in ev_cache:
            ev_cache[key] = jw.evidence(df)
        pr = jw.person_runs(ev_cache[key])
        from v2.segments import union
        out[v] = union([(s, e) for s, e, _ in pr], gap=jw.p["GAP"], min_len=jw.p["MIN_LEN"])
    return out

def sub(vids): return {v: gts[v] for v in vids}

# old rule (current proposals, jaywalking only)
old = {}
for v in LABELS:
    p = json.load(open(f"C:/Users/Lenovo/AppData/Local/Temp/claude/D--hakathon-wiut/7454e6ba-4e25-4cf0-be36-be5d5b30d99e/scratchpad/analysis/prop_{v[:-4]}.json"))
    old[v] = [(s, e) for s, e, l, *_ in p["videos"][v]["proposals"] if l == "jaywalking"]
print("old rule:", round(f1(score(old, gts)), 3), {v: round(f1(score({v: old[v]}, sub([v]))), 3) for v in LABELS})
d = run({}, LABELS)
print("v2 defaults:", round(f1(score(d, gts)), 3), {v: round(f1(score({v: d[v]}, sub([v]))), 3) for v in LABELS})

grid = dict(MARGIN=[0.0, 0.1, 0.25], RATIO=[0.4], PERSON_MIN=[0.5, 1.0], PARTIAL=[0.0, 0.75], KERB=[False, True], GAP=[1.0, 2.0, 4.0, 6.0], MIN_LEN=[0.5, 1.5, 3.0])
combos = [dict(zip(grid, c)) for c in itertools.product(*grid.values())]
results = []
for c in combos:
    per = run(c, LABELS)
    results.append((c, {v: per[v] for v in LABELS}))
def best_on(vids):
    return max(results, key=lambda r: f1(score({v: r[1][v] for v in vids}, sub(vids))))
allbest = best_on(list(LABELS))
print("best on all 3 (optimistic):", allbest[0], round(f1(score(allbest[1], gts)), 3))
lovo = {}
for held in LABELS:
    train = [v for v in LABELS if v != held]
    c, _ = best_on(train)
    per = dict(next(r for r in results if r[0] == c)[1])
    lovo[held] = (c, per[held])
    print(f"  tuned without {held}: {c} -> {held} F1 {f1(score({held: per[held]}, sub([held]))):.3f}")
print("leave-one-video-out pooled:", round(f1(score({v: lovo[v][1] for v in LABELS}, gts)), 3))
r = score(allbest[1], gts)["per_class"]["jaywalking"]
print("best params per tau:", {k: (round(x["f1"], 2), x["tp"], x["fp"], x["fn"]) for k, x in r.items() if k != "f1_mean"})
json.dump({v: allbest[1][v] for v in LABELS}, open(DATA / "runs/jaywalking_v2/best_pred.json", "w"))
