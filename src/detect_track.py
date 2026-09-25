"""Detect and track road users in a video and save the tracks (plus an annotated preview).

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

from tracker import COLUMNS, DEFAULT_TRACKER, DEFAULT_WEIGHTS, iter_tracks, load_model

COLORS = {"person": (0, 0, 255), "bicycle": (255, 128, 0), "car": (0, 200, 0), "motorcycle": (255, 0, 255),
          "bus": (0, 200, 255), "truck": (255, 255, 0)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--imgsz", type=int, default=1280, help="inference size; 4K frames need >=1280 for far pedestrians")
    p.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    p.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    p.add_argument("--duration", type=float, default=0.0, help="seconds to process; 0 = to the end")
    p.add_argument("--tracker", default=str(DEFAULT_TRACKER))
    p.add_argument("--out", default="")
    p.add_argument("--render-width", type=int, default=1280)
    return p.parse_args()


def draw(frame, rows, trails, scale):
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for _, _, tid, cls, _, x1, y1, x2, y2 in rows:
        c = COLORS[cls]
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(small, p1, p2, c, 2)
        cv2.putText(small, f"{cls} #{tid}", (p1[0], max(p1[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
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
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    end = a.start + a.duration if a.duration > 0 else None

    scale = a.render_width / w
    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride,
                             (a.render_width, int(h * scale)))
    trails = defaultdict(lambda: deque(maxlen=30))
    tracks = defaultdict(set)
    frames = []
    t0 = time.time()
    with open(out / "tracks.csv", "w", newline="") as f:
        rows_out = csv.writer(f)
        rows_out.writerow(COLUMNS)
        for k, (idx, t, frame, rows) in enumerate(iter_tracks(video, load_model(a.model), stride=a.stride,
                                                             imgsz=a.imgsz, tracker=a.tracker,
                                                             start_sec=a.start, end_sec=end)):
            for r in rows:
                trails[r[2]].append(((r[5] + r[7]) / 2, r[8]))  # bottom-centre = ground point
                tracks[r[3]].add(r[2])
            rows_out.writerows(rows)
            frames.append(idx)
            vis = draw(frame, rows, trails, scale)
            writer.write(vis)
            if k == 0 or len(frames) % 50 == 0:
                cv2.imwrite(str(out / "snapshot.jpg"), vis)
                print(f"  t={t:6.1f}s  {len(frames)} frames  {len(frames) / (time.time() - t0):.2f} fps processed")
    writer.release()
    el = time.time() - t0
    span = (frames[-1] - frames[0] + 1) / fps if frames else 0
    print(f"done in {el:.0f}s for {span:.0f}s of video ({el / max(span, 1e-9):.2f}x realtime)")
    for cls, ids in tracks.items():
        print(f"  {cls:10s} tracks={len(ids)}")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
