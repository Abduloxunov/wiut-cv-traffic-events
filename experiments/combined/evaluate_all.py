"""Score the whole v2 event layer on the dev labels with the official metric (all classes together), time each class,
and compare with v1 (rules.detect on the same tracks)."""
import json, pickle, time
import pandas as pd
from common import *
from rules import detect as v1_detect
from v2.pipeline import detect_v2
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
V1_ENABLED = ["stopped_vehicle", "jaywalking", "failure_to_yield", "congestion", "wrong_way", "red_light", "stop_line"]
pred1, pred2, timing = {"videos": {}}, {"videos": {}}, {}
for v in LABELS:
    df, scene, _ = data[v]
    sig = pd.read_csv(DATA / f"runs/signals/{v[:-4]}.csv")
    dur = gts[v]["duration"]
    ev1 = v1_detect(df, scene, dur, classes=V1_ENABLED, signals=sig)
    pred1["videos"][v] = {"events": [[s, e, l] for s, e, l, *_ in ev1]}
    tm = {}; t0 = time.time()
    pred2["videos"][v] = {"events": detect_v2(df, scene, dur, signals=sig, timings=tm)}
    timing[v] = (round(time.time() - t0, 1), {k: round(x, 1) for k, x in tm.items()})
for name, pred in [("v1 (current solution)", pred1), ("v2", pred2)]:
    r = ev.evaluate(gts, pred)
    pa = r["part_a"]
    print(f"{name}: Score A = {pa['score_a']:.3f} over {len(pa['classes'])} classes")
    print("   " + ", ".join(f"{c} {x['f1_mean']:.2f}" for c, x in sorted(pa["per_class"].items(), key=lambda kv: -kv[1]["f1_mean"])))
json.dump({"team": "dev-v2", **pred2}, open(DATA / "runs/jaywalking_v2/pred_v2_dev.json", "w"), indent=1)
print("v2 seconds per video (total, per class):", timing)
