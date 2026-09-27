"""Write predictions_samples.json without a GPU: the same steps as solution.py + run_submission.py, in the same format.

  Part A: zones aligned on the first frame, traffic lights read on every 3rd frame, event layer v2 on the tracks of
          YOLO26m @1280 + ByteTrack every 3rd frame (saved by src/detect_track.py with exactly solution.py's settings,
          because tracking 4K on CPU takes ~1 h per video), cleaned with run_submission.clean_events.
  Part B: solution.RiskEstimator fed every frame in order, exactly like run_submission.run_risk (time budget lifted:
          this is CPU; on the T4 the default budget applies).

  RISK_BUDGET=100 python tools/make_predictions_samples.py --videos ../sample_videos --runs ../runs \
      --out predictions_samples.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("RISK_BUDGET", "100")
import run_submission as harness  # noqa: E402
import signals  # noqa: E402
import solution  # noqa: E402
from rules import add_motion  # noqa: E402
from scene import Scene  # noqa: E402
from v2.pipeline import detect_v2  # noqa: E402


def one(path, runs):
    meta = harness.probe(path) if hasattr(harness, "probe") else None
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = meta or {"video_id": path.name, "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)),
                    "n_frames": n, "duration": n / fps}
    ok, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    scene = Scene.for_video(str(path), background=first)
    rects = signals.boxes(scene)
    est = solution.RiskEstimator()
    est.reset({k: meta[k] for k in ("video_id", "fps", "width", "height", "n_frames")})
    curve, lights, idx, last = [], [], 0, 0.0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        last = min(1.0, max(0.0, float(est.step(frame, t))))
        curve.append([round(t, 4), round(last, 4)])
        if idx % solution.STRIDE == 0:
            lights.append({"t_sec": round(t, 3), **signals.states(frame, rects)})
        idx += 1
        if idx % 1500 == 0:
            print(f"  {path.name}: {t:.0f}/{meta['duration']:.0f} s, {time.time() - t0:.0f} s", flush=True)
    cap.release()
    df = add_motion(pd.read_csv(runs / path.stem / "tracks.csv"), fps)
    events = detect_v2(df, scene, meta["duration"], signals=pd.DataFrame(lights), fps=fps)
    events, problems = harness.clean_events(events, solution.CLASSES, meta["duration"])
    return {"events": events, "risk": curve}, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT.parent / "sample_videos"))
    ap.add_argument("--runs", default=str(ROOT.parent / "runs"))
    ap.add_argument("--out", default=str(ROOT / "predictions_samples.json"))
    ap.add_argument("--team", default="wiut-cv-traffic-events")
    a = ap.parse_args()
    result = {"team": a.team, "videos": {}}
    for path in sorted(Path(a.videos).glob("*.MP4")) + sorted(Path(a.videos).glob("*.mp4")):
        if path.name in result["videos"]:
            continue
        entry, problems = one(path, Path(a.runs))
        result["videos"][path.name] = entry
        print(f"[{path.name}] {len(entry['events'])} events, {len(entry['risk'])} risk samples, problems: {problems}")
        Path(a.out).write_text(json.dumps(result, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
