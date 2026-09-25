"""Compare detector / tracker configurations on one clip without labels.

For each config it measures inference speed, detections per frame, and track quality
(how many tracks, how long they live, how many are fragments). Detection quality is
estimated by agreement with the largest model (used as a pseudo ground truth).

Frames are decoded once, downscaled to 1920 wide and cached, so every config sees
exactly the same input and decode cost is excluded from the timings.

Example:
  python src/bench_detectors.py --video samples/C3897.MP4 --start 60 --duration 20
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONFIGS = [  # name, weights, imgsz, tracker
    ("yolo26m@1280+bytetrack", "yolo26m.pt", 1280, "src/bytetrack_tuned.yaml"),  # reference
    ("yolo26s@1280+bytetrack", "yolo26s.pt", 1280, "src/bytetrack_tuned.yaml"),
    ("yolo26s@1280+botsort", "yolo26s.pt", 1280, "botsort.yaml"),
    ("yolo26s@960+bytetrack", "yolo26s.pt", 960, "src/bytetrack_tuned.yaml"),
    ("yolo26n@1280+bytetrack", "yolo26n.pt", 1280, "src/bytetrack_tuned.yaml"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float, default=60.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--out", default="runs/bench")
    return p.parse_args()


def load_frames(path, start, duration, stride, width=1920):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    frames = []
    for i in range(int(duration * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA))
    return frames, fps


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def agreement(pred, ref, group):
    """F1 of boxes vs the reference model (greedy IoU>=0.5), pooled over frames, for one class group."""
    tp = fp = fn = 0
    for p, r in zip(pred, ref):
        pb = np.array([b for b, c in p if c in group]).reshape(-1, 4)
        rb = np.array([b for b, c in r if c in group]).reshape(-1, 4)
        m = iou_matrix(pb, rb)
        matched = 0
        while m.size and m.max() >= 0.5:
            i, j = np.unravel_index(m.argmax(), m.shape)
            matched += 1
            m[i, :] = 0
            m[:, j] = 0
        tp += matched
        fp += len(pb) - matched
        fn += len(rb) - matched
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def run_config(frames, weights, imgsz, tracker):
    model = YOLO(f"weights/{weights}")
    per_frame, tracks = [], {}
    t0 = time.time()
    for k, f in enumerate(frames):
        r = model.track(f, imgsz=imgsz, conf=0.1, classes=list(CLASSES), tracker=tracker, persist=True, verbose=False)[0]
        dets = []
        if r.boxes.id is not None:
            for tid, cls, xyxy in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.xyxy.tolist()):
                dets.append((xyxy, cls))
                tracks.setdefault(tid, []).append(k)
        per_frame.append(dets)
    fps = len(frames) / (time.time() - t0)
    lengths = np.array([len(v) for v in tracks.values()])
    return per_frame, {
        "infer_fps_cpu": round(fps, 2),
        "dets_per_frame": {CLASSES[c]: round(sum(1 for d in per_frame for _, cc in d if cc == c) / len(frames), 1)
                           for c in CLASSES},
        "tracks": int(len(lengths)),
        "median_track_len": float(np.median(lengths)) if len(lengths) else 0.0,
        "fragments_lt5": round(float((lengths < 5).mean()), 3) if len(lengths) else 0.0,
    }


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, fps = load_frames(a.video, a.start, a.duration, a.stride)
    print(f"{len(frames)} frames cached from {Path(a.video).name} ({a.start}-{a.start + a.duration}s, every {a.stride})")
    results, ref = {}, None
    for name, weights, imgsz, tracker in CONFIGS:
        per_frame, stats = run_config(frames, weights, imgsz, tracker)
        if ref is None:
            ref = per_frame
        stats["agree_f1_vehicles"] = round(agreement(per_frame, ref, {2, 3, 5, 7}), 3)
        stats["agree_f1_person"] = round(agreement(per_frame, ref, {0}), 3)
        results[name] = stats
        print(name, json.dumps(stats))
        (out / "bench.json").write_text(json.dumps({"video": Path(a.video).name, "start": a.start,
                                                    "duration": a.duration, "stride": a.stride,
                                                    "results": results}, indent=1))


if __name__ == "__main__":
    main()
