"""Robustness: change one setting at a time around the defaults and re-score (4 labelled events: no fine-tuning)."""
import json, pickle
from common import *
from v2.stopped_vehicle import DEFAULTS, StoppedVehicle
data = pickle.load(open(DATA / "runs/jaywalking_v2/cache.pkl", "rb"))
gts = {v: json.load(open(DATA / "label_bundles" / LABELS[v], encoding="utf-8"))[v] for v in LABELS}
def run(**kw):
    pred = {v: [(s, e) for s, e, _ in StoppedVehicle(data[v][1], **kw).detect(data[v][0], gts[v]["duration"])] for v in LABELS}
    return score(pred, gts, "stopped_vehicle")["per_class"]["stopped_vehicle"]["f1_mean"]
print("defaults", round(run(), 3))
for k, vals in dict(STILL=[0.2, 0.4], LINK_IOU=[0.6, 0.8], MAX_GAP=[5, 15], QUEUE_SHARE=[0.3, 0.7], QUEUE_DIST=[2, 4],
                    QUEUE_SYNC=[4, 10], MAX_QUEUE=[90, 150], GAP=[2, 10], JUNCTION=[True]).items():
    print(f"  {k:12s} default {DEFAULTS[k]!s:6s} -> " + ", ".join(f"{v}: {run(**{k: v}):.3f}" for v in vals))
