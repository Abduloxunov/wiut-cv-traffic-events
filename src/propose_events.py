"""Rule-based event proposals from saved tracks, for the labelling tool.

Output (loads into tools/label_tool.html as dashed pre-labels):
  {"videos": {"C3902.MP4": {"duration": 317.8, "fps": 29.97,
                            "proposals": [[start, end, label, score, why], ...]}}}

Example:
  python src/propose_events.py --run runs/C3902 --video samples/C3902.MP4
"""
import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from align import median_background
from rules import add_motion, detect
from scene import Scene


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder with tracks.csv (and optionally background.jpg)")
    p.add_argument("--video", required=True, help="the video the tracks came from")
    p.add_argument("--out", default="", help="default: <run>/proposals.json")
    return p.parse_args()


def main():
    a = parse_args()
    run = Path(a.run)
    cap = cv2.VideoCapture(a.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n / fps

    bg_path = run / "background.jpg"
    if not bg_path.exists():
        cv2.imwrite(str(bg_path), median_background(a.video))
    scene = Scene.for_video(a.video, background=cv2.imread(str(bg_path)))
    df = add_motion(pd.read_csv(run / "tracks.csv"), fps)
    events = detect(df, scene, duration)

    counts = pd.Series([e[2] for e in events]).value_counts().to_dict() if events else {}
    print(f"{Path(a.video).name}: {len(events)} proposals {counts}")
    out = Path(a.out or run / "proposals.json")
    out.write_text(json.dumps({"videos": {Path(a.video).name: {
        "duration": round(duration, 3), "fps": round(fps, 3),
        "proposals": [[s, e, label, 0.5, why] for s, e, label, why in events]}}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
