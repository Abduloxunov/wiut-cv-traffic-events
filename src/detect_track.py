"""Detect and track road users (cars, trucks, buses, motorbikes, bicycles, people) in a video.

Writes, into --out:
  tracks.csv      one row per detection: frame, t_sec, track_id, cls, conf, x1, y1, x2, y2 (original pixels)
  annotated.mp4   downscaled video with boxes, ids and short trails
  snapshot.jpg    one annotated frame from the middle of the processed range

Example:
  python src/detect_track.py --video samples/C3905.MP4 --duration 30
"""
import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO ids we care about -> short name
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {0: (0, 0, 255), 1: (255, 128, 0), 2: (0, 200, 0), 3: (255, 0, 255), 5: (0, 200, 255), 7: (255, 255, 0)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default="weights/yolo26s.pt")
    p.add_argument("--imgsz", type=int, default=1280, help="inference size; 4K frames need >=1280 for far vehicles")
    p.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    p.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    p.add_argument("--duration", type=float, default=0.0, help="seconds to process; 0 = to the end")
    p.add_argument("--conf", type=float, default=0.1, help="keep low scores so the tracker's second stage can use them")
    p.add_argument("--tracker", default="src/bytetrack_tuned.yaml")
    p.add_argument("--out", default="")
    p.add_argument("--render-width", type=int, default=1280)
    return p.parse_args()


def draw(frame, boxes, trails, scale):
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for tid, cls, conf, (x1, y1, x2, y2) in boxes:
        c = COLORS[cls]
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(small, p1, p2, c, 2)
        cv2.putText(small, f"{CLASSES[cls]} #{tid}", (p1[0], max(p1[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        pts = [(int(x * scale), int(y * scale)) for x, y in trails[tid]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(small, a, b, c, 1, cv2.LINE_AA)
    return small


def main():
    a = parse_args()
    video = Path(a.video)
    out = Path(a.out or f"runs/{video.stem}")
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = int(a.start * fps)
    last = n_frames if a.duration <= 0 else min(n_frames, first + int(a.duration * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    print(f"{video.name}: {w}x{h} @ {fps:.2f} fps, {n_frames} frames; processing {first}..{last} every {a.stride}")

    model = YOLO(a.model)
    scale = a.render_width / w
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / a.stride, (a.render_width, int(h * scale)))
    trails = defaultdict(lambda: deque(maxlen=30))
    counts = defaultdict(int)
    track_ids = defaultdict(set)
    mid = first + (last - first) // 2 // a.stride * a.stride

    t0 = time.time()
    with open(out / "tracks.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])
        for idx in range(first, last):
            if (idx - first) % a.stride:
                cap.grab()  # decode-free skip where the codec allows it
                continue
            ok, frame = cap.read()
            if not ok:
                break
            r = model.track(frame, imgsz=a.imgsz, conf=a.conf, classes=list(CLASSES),
                            tracker=a.tracker, persist=True, verbose=False)[0]
            boxes = []
            if r.boxes.id is not None:
                for tid, cls, conf, xyxy in zip(r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(),
                                                r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    boxes.append((tid, cls, conf, xyxy))
                    trails[tid].append(((xyxy[0] + xyxy[2]) / 2, xyxy[3]))  # bottom-centre = ground point
                    counts[cls] += 1
                    track_ids[cls].add(tid)
                    rows.writerow([idx, round(idx / fps, 3), tid, CLASSES[cls], round(conf, 3),
                                   *(round(v, 1) for v in xyxy)])
            vis = draw(frame, boxes, trails, scale)
            writer.write(vis)
            if idx == mid:
                cv2.imwrite(str(out / "snapshot.jpg"), vis)
            done = (idx - first) // a.stride + 1
            if done % 25 == 0:
                el = time.time() - t0
                print(f"  t={idx / fps:6.1f}s  {done} frames  {done / el:.2f} fps processed")

    writer.release()
    el = time.time() - t0
    processed_sec = (last - first) / fps
    print(f"done in {el:.0f}s for {processed_sec:.0f}s of video ({el / processed_sec:.2f}x realtime)")
    for cls, name in CLASSES.items():
        print(f"  {name:10s} detections={counts[cls]:6d}  unique tracks={len(track_ids[cls])}")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
