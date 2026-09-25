"""Replay saved tracks through the causal risk estimator (src/risk.py) and measure false alarms.

The samples contain no accidents, so every alarm here is a false alarm (or a near miss worth a look).
Alarms are counted exactly like evaluate.py: runs of score >= 0.5, runs closer than 2 s merged.

  python experiments/risk_on_tracks.py ../runs/C3897 ../runs/C3902 ../runs/C3905
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from risk import CollisionRisk  # noqa: E402
from scene import Scene  # noqa: E402
import cv2  # noqa: E402

THETA, MERGE_GAP = 0.5, 2.0


def alarms(times, scores):
    runs, start, end = [], None, None
    for t, s in zip(times, scores):
        if s >= THETA:
            start = t if start is None else start
            end = t
        elif start is not None:
            runs.append([start, end])
            start = None
    if start is not None:
        runs.append([start, end])
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def main():
    total_sec, total_alarms, summary = 0.0, 0, {}
    for run in map(Path, sys.argv[1:]):
        df = pd.read_csv(run / "tracks.csv")
        df["cls"] = df.groupby("track_id").cls.transform(lambda s: s.mode().iloc[0])
        scene = Scene.for_video(f"../sample_videos/{run.name}.MP4", background=cv2.imread(str(run / "background.jpg")))
        est, times, scores, peaks = CollisionRisk(scene), [], [], []
        for (frame, t), g in df.groupby(["frame", "t_sec"]):
            r, best = est.update(t, g.values.tolist())
            times.append(t)
            scores.append(r)
            if best:
                peaks.append((r, t, best))
        a = alarms(times, scores)
        dur = times[-1] - times[0]
        total_sec += dur
        total_alarms += len(a)
        sc = np.array(scores)
        top = sorted(peaks, key=lambda x: -x[0])[:5]
        summary[run.name] = {"duration_s": round(dur, 1), "alarms": a,
                             "p50": round(float(np.percentile(sc, 50)), 4), "p99": round(float(np.percentile(sc, 99)), 3),
                             "max": round(float(sc.max()), 3),
                             "top": [(round(r, 3), round(t, 1), b) for r, t, b in top]}
        pd.DataFrame({"t_sec": times, "risk": scores}).to_csv(run / "risk.csv", index=False)
        print(f"{run.name}: {len(a)} alarms in {dur:.0f}s  p50={summary[run.name]['p50']} "
              f"p99={summary[run.name]['p99']} max={summary[run.name]['max']}")
        for r, t, b in top:
            print(f"   peak {r:.3f} at {t:6.1f}s  {b}")
    print(f"TOTAL: {total_alarms} false alarms in {total_sec / 60:.1f} min "
          f"({total_alarms / (total_sec / 3600):.1f} per hour)")
    Path("experiments/data/risk_summary.json").write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
