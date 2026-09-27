"""stop_line v1 vs v2 on the dev labels."""
import json, pickle
import pandas as pd
from common import *
from v2.stop_line import StopLine
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "stop_line"; fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"
res = lambda p: score(p, gts, L)["per_class"][L]
old = {}
for v in LABELS:
    pr = json.load(open(f"C:/Users/Lenovo/AppData/Local/Temp/claude/D--hakathon-wiut/7454e6ba-4e25-4cf0-be36-be5d5b30d99e/scratchpad/analysis/prop_{v[:-4]}.json"))
    old[v] = [(s, e) for s, e, l, *_ in pr["videos"][v]["proposals"] if l == L]
r = res(old); print("v1:", round(r["f1_mean"], 3), {k: (x["tp"], x["fp"], x["fn"]) for k, x in r.items() if k != "f1_mean"})
for arrive in ["red", "not_green"]:
    pred = {v: [(s, e) for s, e, _ in StopLine(data[v][1], pd.read_csv(DATA / f"runs/signals/{v[:-4]}.csv"), ARRIVE=arrive).detect(data[v][0])] for v in LABELS}
    r = res(pred); print(f"v2 ARRIVE={arrive}:", round(r["f1_mean"], 3), {k: (x["tp"], x["fp"], x["fn"]) for k, x in r.items() if k != "f1_mean"})
    for v in LABELS:
        print(f"   {v} labels {[f'{fmt(s)}-{fmt(e)}' for s, e, l in gts[v]['events'] if l == L]}  v1 {[f'{fmt(s)}-{fmt(e)}' for s, e in old[v]]}  v2 {[f'{fmt(s)}-{fmt(e)}' for s, e in pred[v]]}")
