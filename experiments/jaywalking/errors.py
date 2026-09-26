import json, sys
from common import *
best = json.load(open(DATA / "runs/jaywalking_v2/best_pred.json"))
def tiou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u
for v in LABELS:
    g = [e[:2] for e in json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v]["events"] if e[2] == "jaywalking"]
    p = best[v]
    print(v)
    for s, e in sorted(g):
        b = max(p, key=lambda x: tiou((s, e), x), default=None)
        print(f"   GT {s:6.1f}-{e:6.1f} ({e-s:5.1f}s)  best pred {b[0]:6.1f}-{b[1]:6.1f} tIoU {tiou((s,e),b):.2f}" if b else f"   GT {s:6.1f}-{e:6.1f} none")
    for s, e in p:
        if max((tiou((s, e), x) for x in g), default=0) < 0.3:
            print(f"   FP {s:6.1f}-{e:6.1f} ({e-s:5.1f}s)")
