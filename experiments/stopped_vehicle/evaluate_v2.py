"""Score stopped_vehicle v1 (old rule) and v2 on the dev labels."""
import json, pickle
from common import *
from v2.stopped_vehicle import StoppedVehicle

data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
f1 = lambda p: score(p, gts, "stopped_vehicle")["per_class"]["stopped_vehicle"]
fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"
old = {}
for v in LABELS:
    p = json.load(open(f"C:/Users/Lenovo/AppData/Local/Temp/claude/D--hakathon-wiut/7454e6ba-4e25-4cf0-be36-be5d5b30d99e/scratchpad/analysis/prop_{v[:-4]}.json"))
    old[v] = [(s, e) for s, e, l, *_ in p["videos"][v]["proposals"] if l == "stopped_vehicle"]
print("old rule:", round(f1(old)["f1_mean"], 3))
pred = {}
for v in LABELS:
    df, scene, _ = data[v]
    pred[v] = [(s, e) for s, e, _ in StoppedVehicle(scene).detect(df, gts[v]["duration"])]
    g = [e[:2] for e in gts[v]["events"] if e[2] == "stopped_vehicle"]
    print(f"  {v}: labels {[f'{fmt(s)}-{fmt(e)}' for s, e in g]}\n           v2     {[f'{fmt(s)}-{fmt(e)}' for s, e in pred[v]]}")
r = f1(pred); print("v2:", round(r["f1_mean"], 3), {k: (round(x["f1"], 2), x["tp"], x["fp"], x["fn"]) for k, x in r.items() if k != "f1_mean"})
