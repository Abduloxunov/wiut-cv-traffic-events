"""illegal_turn v2 on the dev labels (2 labelled events) with a small timing sweep."""
import json, pickle
from common import *
from v2.illegal_turn import IllegalTurn
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
L = "illegal_turn"; fmt = lambda t: f"{int(t//60)}:{t%60:04.1f}"
res = lambda pred: score(pred, gts, L)["per_class"][L]
for pre in [0.5, 1.5, 2.5]:
    for post in [0.0, 0.5, 1.5, 3.0]:
        for near in [200, 400, 800]:
            pred = {v: [(s, e) for s, e, _ in IllegalTurn(data[v][1], PRE=pre, POST=post, TO_NEAR=near).detect(data[v][0])] for v in LABELS}
            r = res(pred)
            print(f"PRE {pre} POST {post} NEAR {near}: {r['f1_mean']:.3f} ", {k: (x['tp'], x['fp'], x['fn']) for k, x in r.items() if k != 'f1_mean'}, [f"{fmt(s)}-{fmt(e)}" for v in LABELS for s, e in pred[v]])
