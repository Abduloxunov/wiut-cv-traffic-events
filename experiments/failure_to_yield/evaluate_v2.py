"""Score failure_to_yield v1 and v2 on the dev labels; list matches."""
import json, pickle, sys
from common import *
from v2.failure_to_yield import DEFAULTS, FailureToYield
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "failure_to_yield"
res = lambda p: score(p, gts, L)["per_class"][L]
fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"
old = {}
for v in LABELS:
    pr = json.load(open(f"C:/Users/Lenovo/AppData/Local/Temp/claude/D--hakathon-wiut/7454e6ba-4e25-4cf0-be36-be5d5b30d99e/scratchpad/analysis/prop_{v[:-4]}.json"))
    old[v] = [(s, e) for s, e, l, *_ in pr["videos"][v]["proposals"] if l == L]
r = res(old); print("old rule:", round(r["f1_mean"], 3), {k: (x["tp"], x["fp"], x["fn"]) for k, x in r.items() if k != "f1_mean"})
def run(**kw):
    return {v: [(s, e) for s, e, _ in FailureToYield(data[v][1], **kw).detect(data[v][0])] for v in LABELS}
if __name__ == "__main__":
    pred = run()
    r = res(pred); print("v2:", round(r["f1_mean"], 3), {k: (round(x["f1"], 2), x["tp"], x["fp"], x["fn"]) for k, x in r.items() if k != "f1_mean"})
    for v in LABELS:
        g = [e[:2] for e in gts[v]["events"] if e[2] == L]
        print(f"  {v}: labels {', '.join(f'{fmt(s)}-{fmt(e)}' for s, e in g)}")
        print(f"         v2 {len(pred[v])}: {', '.join(f'{fmt(s)}-{fmt(e)}' for s, e in pred[v])}")
