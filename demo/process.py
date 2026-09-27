"""Demo pipeline for the website: the same stages as solution.py, sized for a CPU server.

  video -> first frame -> zones aligned (SIFT homography) -> FFmpeg decodes TARGET_HZ frames per second scaled to
  DECODE_W wide (any input size; multi-threaded, so a 4K upload costs little) -> YOLO26s/n @960 + ByteTrack
  (boxes mapped to the 3840x2160 frame the zones and rules are calibrated in) -> traffic-light states ->
  event layer v2 -> causal risk replay -> annotated playback + events + risk curve

Differences from the submission (solution.py), stated on the demo page: a smaller detector (YOLO26s for clips up to
SMALL_MAX_FRAMES processed frames, YOLO26n for longer ones, instead of YOLO26m) at 960 px and ~5 processed frames per
second instead of 10, so a 2-minute clip finishes in minutes on 2 CPU cores.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import signals  # noqa: E402
from risk import CollisionRisk  # noqa: E402
from rules import add_motion  # noqa: E402
from scene import Scene  # noqa: E402
from tracker import COLUMNS, load_model, track_frame  # noqa: E402
from v2.pipeline import detect_v2  # noqa: E402

CANON = (3840, 2160)          # the zones and every rule threshold live in the camera's native 4K frame
MAX_SEC = 120.0               # longer uploads are cut to the first 2 minutes
TARGET_HZ = 5.0               # processed frames per second
IMGSZ = 960
WEIGHTS = {"s": ROOT / "weights" / "yolo26s.pt", "n": ROOT / "weights" / "yolo26n.pt"}
SMALL_MAX_FRAMES = 250        # ~50 s at 5 Hz; longer clips use the nano detector (about 3x faster on CPU)
OUT_WIDTH = 960
DECODE_W = 1920               # decoded width: enough for the detector at 960 and for the small signal heads

EVENT_COLORS = {  # BGR, same palette as tools/render_results.py
    "jaywalking": (94, 197, 34), "stopped_vehicle": (8, 179, 234), "failure_to_yield": (166, 184, 20),
    "congestion": (246, 130, 59), "stop_line": (133, 113, 251), "solid_line_crossing": (21, 204, 250),
    "illegal_turn": (241, 102, 99),
}
BOX_COLORS = {"person": (80, 200, 255), "car": (255, 170, 60), "bus": (60, 220, 120), "truck": (200, 120, 255),
              "motorcycle": (0, 140, 255), "bicycle": (255, 255, 0)}

_models = {}


def model(size):
    if size not in _models:
        _models[size] = load_model(WEIGHTS[size])
    return _models[size]


def run(video_path, out_dir, progress=lambda frac, msg: None):
    """Process one uploaded video. Returns a dict with events, risk curve, paths and notes."""
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not read this file as a video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    notes = []
    duration_full = n_total / fps if n_total else 0.0
    n = min(n_total, int(MAX_SEC * fps)) if n_total else int(MAX_SEC * fps)
    if duration_full > MAX_SEC:
        notes.append(f"Video is {duration_full:.0f} s long; the first {MAX_SEC:.0f} s were processed.")
    hz = min(TARGET_HZ, fps)
    dw = min(DECODE_W, W) // 2 * 2
    dh = int(round(H * dw / W / 2)) * 2
    sx, sy = CANON[0] / dw, CANON[1] / dh

    ok, first = cap.read()
    if not ok:
        raise ValueError("The video has no readable frames.")
    progress(0.02, "Aligning the scene zones to this video")
    canon_first = cv2.resize(first, CANON, interpolation=cv2.INTER_LINEAR)
    scene = Scene.from_image(canon_first)
    if getattr(scene, "aligned", True) is False:
        notes.append("Could not match this view to our camera; zones are only scaled, so events may be wrong.")
    _zones_preview(first, scene, CANON[0] / W, CANON[1] / H, out_dir / "zones.jpg")
    rects = signals.boxes(scene)
    rects_small = {k: (int(x1 / sx), int(y1 / sy), int(np.ceil(x2 / sx)), int(np.ceil(y2 / sy)))
                   for k, (x1, y1, x2, y2) in rects.items()}

    cap.release()
    n_proc = int((n / fps) * hz)
    size = "s" if n_proc <= SMALL_MAX_FRAMES else "n"
    notes.append(f"Input {W}x{H} at {fps:.2f} fps, decoded at {dw}x{dh}, {hz:.0f} frames per second; "
                 f"detector YOLO26{size} at {IMGSZ} px.")
    m = model(size)
    m.predictor = None
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="bgr24", input_params=["-t", f"{n / fps:.3f}"],
                                        output_params=["-vf", f"fps={hz},scale={dw}:{dh}"])
    reader.__next__()  # metadata
    rows, light_rows, kept = [], [], []
    for k, raw in enumerate(reader):
        frame = np.frombuffer(raw, np.uint8).reshape(dh, dw, 3).copy()
        t = k / hz
        idx = int(round(t * fps))
        r = track_frame(m, frame, idx, t, imgsz=IMGSZ)
        rows += [[f, tt, tid, c, cf, x1 * sx, y1 * sy, x2 * sx, y2 * sy] for f, tt, tid, c, cf, x1, y1, x2, y2 in r]
        light_rows.append({"t_sec": round(t, 3), **signals.states(frame, rects_small)})
        small = cv2.resize(frame, (OUT_WIDTH, int(dh * OUT_WIDTH / dw)))
        kept.append((idx, t, small, r))
        if len(kept) % 10 == 0:
            progress(0.05 + 0.75 * min(1.0, k / max(n_proc, 1)), f"Detecting and tracking: {t:.0f} / {n / fps:.0f} s")
    duration = n / fps

    progress(0.82, "Finding events")
    df = pd.DataFrame(rows, columns=COLUMNS)
    events = detect_v2(add_motion(df, fps), scene, duration, signals=pd.DataFrame(light_rows), fps=fps) if len(df) else []

    progress(0.86, "Scoring accident risk (Part B)")
    est = CollisionRisk(scene)
    risk = []
    for f, g in df.groupby("frame", sort=True) if len(df) else []:
        tt = float(g.t_sec.iloc[0])
        sc, _ = est.update(tt, g.itertuples(index=False, name=None))
        risk.append([round(tt, 2), round(sc, 4)])

    progress(0.9, "Rendering the annotated playback")
    video_out = out_dir / "annotated.mp4"
    _render(kept, events, risk, duration, hz, dw, video_out)
    counts = {}
    for _, _, lab in events:
        counts[lab] = counts.get(lab, 0) + 1
    result = {"duration": round(duration, 2), "fps": fps, "processed_frames": len(kept), "events": events,
              "counts": counts, "risk": risk, "notes": notes, "seconds": round(time.time() - t_start, 1),
              "aligned": bool(getattr(scene, "aligned", True)), "resolution": f"{W}x{H}"}
    (out_dir / "events.json").write_text(json.dumps(result, indent=1))
    progress(1.0, "Done")
    return result, video_out, out_dir / "events.json"


ZONE_COLORS = {"crosswalk": (255, 255, 255), "island": (120, 120, 255), "sidewalk": (180, 180, 180),
               "stop_line": (60, 60, 255), "solid_line": (0, 220, 255), "approach": (255, 160, 60),
               "intersection": (80, 255, 80), "bus_stop": (255, 80, 255), "parking": (200, 120, 0)}


def _zones_preview(frame, scene, sx, sy, path):
    """The uploaded video's first frame with our zones as aligned to it (proof that the scene map fits)."""
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (OUT_WIDTH, int(h * OUT_WIDTH / w)))
    k = OUT_WIDTH / w
    over = img.copy()
    for z in scene.shapes:
        c = ZONE_COLORS.get(z["type"])
        if c is None:
            continue
        pts = (z["points"] / [sx, sy] * k).astype(np.int32)
        closed = z["type"] not in ("stop_line", "solid_line")
        if closed and z["type"] in ("crosswalk", "island", "intersection"):
            cv2.fillPoly(over, [pts], c)
        cv2.polylines(img, [pts], closed, c, 2, cv2.LINE_AA)
    img = cv2.addWeighted(over, 0.25, img, 0.75, 0)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _render(kept, events, risk, duration, out_fps, W_in, path):
    import imageio_ffmpeg
    if not kept:
        return
    h, w = kept[0][2].shape[:2]
    strip = 90
    s = OUT_WIDTH / W_in
    writer = imageio_ffmpeg.write_frames(str(path), (w, h + strip), fps=out_fps, codec="libx264",
                                         macro_block_size=1, output_params=["-crf", "28", "-preset", "veryfast"])
    writer.send(None)
    labels = sorted({e[2] for e in events})
    rt = np.array([r[0] for r in risk]) if risk else np.zeros(0)
    rv = np.array([r[1] for r in risk]) if risk else np.zeros(0)
    for idx, t, img, rows in kept:
        img = img.copy()
        for _, _, tid, c, _, x1, y1, x2, y2 in rows:
            cv2.rectangle(img, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), BOX_COLORS.get(c, (200, 200, 200)), 1)
        y = 24
        for lab in [e[2] for e in events if e[0] <= t <= e[1]]:
            cv2.putText(img, lab.replace("_", " "), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, lab.replace("_", " "), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EVENT_COLORS.get(lab, (255, 255, 255)), 2, cv2.LINE_AA)
            y += 24
        j = max(0, np.searchsorted(rt, t, side="right") - 1) if len(rt) else 0
        cv2.putText(img, f"{int(t // 60)}:{t % 60:04.1f}  risk {float(rv[j]) if len(rv) else 0:.2f}", (w - 190, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.full((h + strip, w, 3), 25, np.uint8)
        canvas[:h] = img
        row = max(8, (strip - 6) // max(1, len(labels)))
        for i, lab in enumerate(labels):
            yy = h + 3 + i * row
            cv2.putText(canvas, lab, (4, yy + row - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
            for s0, e0, l in events:
                if l == lab:
                    a = 120 + int((w - 130) * s0 / duration)
                    b = 120 + int((w - 130) * e0 / duration)
                    cv2.rectangle(canvas, (a, yy + 1), (max(b, a + 2), yy + row - 2), EVENT_COLORS.get(lab, (200, 200, 200)), -1)
        px = 120 + int((w - 130) * t / duration)
        cv2.line(canvas, (px, h), (px, h + strip), (255, 255, 255), 1)
        writer.send(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes())
    writer.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="demo_out")
    a = p.parse_args()
    res, vid, js = run(a.video, a.out, progress=lambda f, m: print(f"{f:5.0%} {m}", flush=True))
    print(json.dumps({k: v for k, v in res.items() if k != "risk"}, indent=1)[:1500])
