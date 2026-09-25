"""Train the accident verifier head on extracted clip features (from accident_features.py).

Grouped cross-validation by source video (no leakage between windows of one video), reports ROC-AUC and AP,
then fits on everything and saves the head for the pipeline.

  python train/train_accident_head.py --feats data/feats_videomae.npz --out weights/accident_head.joblib
"""
import argparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True)
    p.add_argument("--out", default="weights/accident_head.joblib")
    a = p.parse_args()

    d = np.load(a.feats, allow_pickle=True)
    X, y, groups = d["X"], d["y"], d["video"]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        oof[te] = make_head().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f"grouped 5-fold: ROC-AUC {roc_auc_score(y, oof):.3f}  AP {average_precision_score(y, oof):.3f}  "
          f"(positives {int(y.sum())} / {len(y)})")
    joblib.dump(make_head().fit(X, y), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
