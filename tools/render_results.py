"""Render the website's result assets for each sample video from its saved tracks (same detector, tracker and event
layer as solution.py):

  <out>/<video>_annotated.mp4   1280-wide playback: boxes by class, active-event banner, event timeline strip, playhead
  <out>/<video>.json            {"duration", "fps", "events": [[start, end, label]], "risk": [[t, score]]}

Part B risk here = the causal estimator (src/risk.py) replayed on the saved Part A tracks, one value per processed
frame (the harness writes one per frame from RiskEstimator, which runs its own lighter detector).

  python tools/render_results.py --videos C3897 C3902 C3905 C3896 --out ../website_assets
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from risk import CollisionRisk  # noqa: E402
from rules import add_motion  # noqa: E402
from scene import Scene  # noqa: E402
from v2.pipeline import detect_v2  # noqa: E402

COLORS = {  # BGR
    "person": (80, 200, 255), "car": (255, 170, 60), "bus": (60, 220, 120), "truck": (200, 120, 255),
    "motorcycle": (0, 140, 255), "bicycle": (255, 255, 0),
}
EVENT_COLORS = {
    "jaywalking": (94, 197, 34), "stopped_vehicle": (8, 179, 234), "failure_to_yield": (166, 184, 20),
    "congestion": (246, 130, 59), "stop_line": (133, 113, 251), "solid_line_crossing": (21, 204, 250),
    "illegal_turn": (241, 102, 99), "near_miss": (22, 115, 249), "accident": (68, 68, 239),
}


def risk_curve(df, scene):
    est = CollisionRisk(scene)
    cols = ["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]
    out = []
    for _, g in df[cols].groupby("frame", sort=True):
        t = float(g.t_sec.iloc[0])
        r, _ = est.update(t, g.itertuples(index=False, name=None))
        out.append([round(t, 2), round(r, 4)])
    return out


def draw_timeline(img, events, duration, t, y0, h):
    W = img.shape[1]
    labels = sorted({e[2] for e in events})
    cv2.rectangle(img, (0, y0), (W, y0 + h), (25, 25, 25), -1)
    row = max(8, (h - 6) // max(1, len(labels)))
    for i, lab in enumerate(labels):
        y = y0 + 3 + i * row
        cv2.putText(img, lab, (4, y + row - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 200), 1, cv2.LINE_AA)
        for s, e, l in events:
            if l == lab:
                x0, x1 = 150 + int((W - 160) * s / duration), 150 + int((W - 160) * e / duration)
                cv2.rectangle(img, (x0, y + 1), (max(x1, x0 + 2), y + row - 2), EVENT_COLORS.get(lab, (200, 200, 200)), -1)
    px = 150 + int((W - 160) * t / duration)
    cv2.line(img, (px, y0), (px, y0 + h), (255, 255, 255), 1)


def render(stem, proxies, runs, out_dir, fps=29.97):
    df = add_motion(pd.read_csv(runs / stem / "tracks.csv"), fps)
    scene = Scene.from_image(cv2.imread(str(runs / stem / "background.jpg")))
    sig_path = runs / "signals" / f"{stem}.csv"
    signals = pd.read_csv(sig_path) if sig_path.exists() else None
    cap = cv2.VideoCapture(str(proxies / f"{stem}.MP4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n / fps
    events = detect_v2(df, scene, duration, signals=signals, fps=fps)
    risk = risk_curve(df, scene)
    json.dump({"video": f"{stem}.MP4", "duration": round(duration, 3), "fps": fps, "events": events, "risk": risk},
              open(out_dir / f"{stem}.json", "w"), indent=0)

    sx, sy = W / 3840, H / 2160
    by_frame = {f: g for f, g in df.groupby("frame")}
    frames_sorted = np.array(sorted(by_frame))
    strip = 110
    writer = imageio_ffmpeg.write_frames(str(out_dir / f"{stem}_annotated.mp4"), (W, H + strip), fps=fps,
                                         codec="libx264", quality=None, bitrate=None,
                                         output_params=["-crf", "27", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    writer.send(None)
    risk_t = np.array([r[0] for r in risk]); risk_v = np.array([r[1] for r in risk])
    for i in range(n):
        ok, img = cap.read()
        if not ok:
            break
        t = i / fps
        k = frames_sorted[max(0, np.searchsorted(frames_sorted, i, side="right") - 1)] if len(frames_sorted) else None
        if k is not None and i - k < 3:
            for r in by_frame[k].itertuples():
                c = COLORS.get(r.cls, (200, 200, 200))
                cv2.rectangle(img, (int(r.x1 * sx), int(r.y1 * sy)), (int(r.x2 * sx), int(r.y2 * sy)), c, 1)
        active = [l for s, e, l in events if s <= t <= e]
        y = 26
        for lab in active:
            cv2.putText(img, lab.replace("_", " "), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, lab.replace("_", " "), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EVENT_COLORS.get(lab, (255, 255, 255)), 2, cv2.LINE_AA)
            y += 28
        j = max(0, np.searchsorted(risk_t, t, side="right") - 1) if len(risk_t) else 0
        rv = float(risk_v[j]) if len(risk_v) else 0.0
        cv2.putText(img, f"{stem}  {int(t // 60)}:{t % 60:04.1f}   risk {rv:.2f}", (W - 330, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.zeros((H + strip, W, 3), np.uint8)
        canvas[:H] = img
        draw_timeline(canvas, events, duration, t, H, strip)
        writer.send(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).tobytes())
    writer.close()
    cap.release()
    print(stem, len(events), "events", "max risk %.2f" % (max(risk_v) if len(risk_v) else 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", default=["C3897", "C3902", "C3905", "C3896"])
    p.add_argument("--proxies", default=str(ROOT.parent / "proxies"))
    p.add_argument("--runs", default=str(ROOT.parent / "runs"))
    p.add_argument("--out", default=str(ROOT.parent / "website_assets"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        render(v, Path(a.proxies), Path(a.runs), out)


if __name__ == "__main__":
    main()
