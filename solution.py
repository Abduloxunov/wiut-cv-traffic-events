"""
solution.py — the interface the organizers' harness imports (run_submission.py).

Part A: detect_events(video_path) -> [[start_sec, end_sec, label], ...]
    YOLO26 + ByteTrack over every 3rd frame -> tracks; hand-drawn scene zones aligned to this video
    (SIFT homography against the reference background) -> rule-based events per class.
Part B: RiskEstimator — causal accident risk per frame (not implemented yet: returns 0).

All heavy code lives in src/; see README.md.
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]
RISK_HORIZON_SEC = 5.0

# Classes we currently emit. Emitting a class that is absent from the test set costs a zero in the
# macro average, so a class is only listed here once it is reliable on our dev labels.
ENABLED = ("stopped_vehicle", "jaywalking", "failure_to_yield", "congestion", "wrong_way")
STRIDE = 3                 # process every 3rd frame (10 per second at 30 fps)
IMGSZ = 1280               # detector input width; smaller loses far pedestrians
# x duration for tracking; the harness decodes 4K for Part B at ~1.0-1.3x and rules + alignment take the rest.
# Override only for local CPU experiments, e.g. PART_A_BUDGET=100.
PART_A_BUDGET = float(os.environ.get("PART_A_BUDGET", 1.3))
BACKGROUND_FRAMES = 15     # frames kept for the median background used to align the zones
SEED = 0

_model = None


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_model():
    global _model
    if _model is None:
        from tracker import load_model
        _model = load_model()
    return _model


def detect_events(video_path: str) -> list[list]:
    """Part A. Return [[start_sec, end_sec, label], ...] for one .mp4."""
    import cv2
    import pandas as pd

    from align import median_image
    from rules import add_motion, detect
    from scene import Scene
    from tracker import COLUMNS, iter_tracks

    t0 = time.perf_counter()
    _seed_everything()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n_frames / fps
    deadline = t0 + PART_A_BUDGET * duration

    # one pass: tracking, and a few full frames kept for the background (no second decode)
    keep_every = max(1, n_frames // STRIDE // BACKGROUND_FRAMES)
    rows, bg_frames = [], []
    for k, (_, _, frame, frame_rows) in enumerate(iter_tracks(video_path, _get_model(), stride=STRIDE,
                                                              imgsz=IMGSZ, deadline=deadline)):
        rows.extend(frame_rows)
        if k % keep_every == 0 and len(bg_frames) < BACKGROUND_FRAMES:
            bg_frames.append(frame)
    if not bg_frames:
        return []
    background = median_image(bg_frames)
    del bg_frames

    scene = Scene.for_video(video_path, background=background)
    df = add_motion(pd.DataFrame(rows, columns=COLUMNS), fps)
    events = detect(df, scene, duration, classes=ENABLED)
    print(f"[solution] {Path(video_path).name}: {len(rows)} detections, {len(events)} events, "
          f"{time.perf_counter() - t0:.0f}s", file=sys.stderr)
    return [[s, e, label] for s, e, label, _ in events]


class RiskEstimator:
    """Part B (optional). Causal: step() sees frames in order and nothing else."""

    def reset(self, meta: dict) -> None:
        # meta = {"video_id", "fps", "width", "height", "n_frames"}
        self.meta = meta
        self.last_score = 0.0

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        # frame: BGR uint8 (H, W, 3). Return P(accident starts within 5 s) in [0, 1].
        return self.last_score
