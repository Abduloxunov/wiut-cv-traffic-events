"""Detect and track road users through a video: the shared first stage of every pipeline.

Yields one record per processed frame so callers can render, stop early on a time budget,
or just collect a table of detections.
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("YOLO_OFFLINE", "1")  # never reach for the internet during evaluation
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_WEIGHTS = ROOT / "weights" / "yolo26m.pt"
DEFAULT_TRACKER = ROOT / "src" / "bytetrack_tuned.yaml"
COLUMNS = ["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]


def load_model(weights=DEFAULT_WEIGHTS):
    return YOLO(str(weights))


def device_args():
    """Half precision on GPU, full on CPU."""
    import torch
    return {"device": 0, "half": True} if torch.cuda.is_available() else {"device": "cpu"}


def track_frame(model, frame, idx, t, imgsz=1280, conf=0.1, tracker=DEFAULT_TRACKER):
    """Detect + track one frame (tracker state persists in `model`). Rows as in iter_tracks."""
    r = model.track(frame, imgsz=imgsz, conf=conf, classes=list(COCO_CLASSES), tracker=str(tracker),
                    persist=True, verbose=False, **device_args())[0]
    if r.boxes.id is None:
        return []
    ids, cls, confs = r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()
    boxes = np.round(r.boxes.xyxy.cpu().numpy(), 1).tolist()
    return [[idx, round(t, 3), tid, COCO_CLASSES[c], round(cf, 3), *b] for tid, c, cf, b in zip(ids, cls, confs, boxes)]


def iter_tracks(video_path, model, stride=3, imgsz=1280, conf=0.1, tracker=DEFAULT_TRACKER,
                start_sec=0.0, end_sec=None, deadline=None):
    """Yield (frame_idx, t_sec, frame, rows) for every processed frame.

    rows: list of [frame, t_sec, track_id, cls_name, conf, x1, y1, x2, y2] in original pixels.
    Stops early (without error) when time.perf_counter() passes `deadline`.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(start_sec * fps)
    last = n if end_sec is None else min(n, int(end_sec * fps))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    model.predictor = None  # fresh tracker state for every video
    try:
        for idx in range(first, last):
            if (idx - first) % stride:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok or (deadline is not None and time.perf_counter() > deadline):
                break
            yield idx, idx / fps, frame, track_frame(model, frame, idx, idx / fps, imgsz, conf, tracker)
    finally:
        cap.release()


def track_video(video_path, model, **kwargs):
    """Run iter_tracks to the end and return all detections as a DataFrame."""
    rows = [row for _, _, _, frame_rows in iter_tracks(video_path, model, **kwargs) for row in frame_rows]
    return pd.DataFrame(rows, columns=COLUMNS)
