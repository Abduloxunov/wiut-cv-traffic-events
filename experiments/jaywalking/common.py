"""Shared loading and scoring for the jaywalking-from-scratch experiments (dev labels = our 3 labelled videos)."""
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import evaluate as ev  # noqa: E402
from rules import add_motion  # noqa: E402
from scene import Scene  # noqa: E402

DATA = ROOT.parent
LABELS = {"C3897.MP4": "labels_C3897 v2 merged.json", "C3902.MP4": "labels_C3902 done.json",
          "C3905.MP4": "labels_C3905 done.json"}
FPS = 29.97


def load(vid):
    """tracks (with motion columns), aligned scene, ground truth entry for one labelled video."""
    stem = vid[:-4]
    df = add_motion(pd.read_csv(DATA / "runs" / stem / "tracks.csv"), FPS)
    scene = Scene.from_image(cv2.imread(str(DATA / "runs" / stem / "background.jpg")))
    gt = json.load(open(DATA / "label_bundles" / LABELS[vid], encoding="utf-8"))[vid]
    return df, scene, gt


def score(pred_by_vid, gts, label="jaywalking"):
    """Official metric restricted to one class: returns (mean F1, {tau: (f1, tp, fp, fn)})."""
    gt = {v: {**g, "events": [e for e in g["events"] if e[2] == label]} for v, g in gts.items()}
    pred = {"videos": {v: {"events": [[s, e, label] for s, e in segs]} for v, segs in pred_by_vid.items()}}
    a = ev.evaluate(gt, pred)["part_a"]
    return a
