"""Traffic-light state from the drawn signal boxes, by lamp colour.

Only heads whose lit face points at the camera are readable (signal_4 pedestrian, signal_5 vehicle in our
zones); the gantry heads show their backs. Lit lamps are much dimmer in daylight (V ~100-125) than at dusk (255), and an unlit red lens
still glows faintly, so a fixed brightness threshold fails. Instead each colour band is scored by the brightness
of its brightest saturated pixels, and the lit lamp must beat the other colours by a clear margin.

CLI: writes a per-time state table and a timeline plot for one video.
  python src/signals.py --video samples/C3905.MP4 --every 0.5
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

READABLE = ("signal_4", "signal_5")
# OpenCV hue is 0..179. Green lamps look cyan at dusk, so the green band is wide.
BANDS = {"red": [(0, 10), (165, 180)], "yellow": [(11, 32)], "green": [(45, 100)]}
MIN_SAT = 60        # lamp pixels are coloured, housings are grey
TOP_K = 20          # brightness of a band = mean V of its TOP_K brightest saturated pixels
MIN_SCORE = 60      # below this nothing is lit
MARGIN = 15         # the lit colour must be this much brighter than the next one


def boxes(scene, names=READABLE):
    """Pixel rectangles (x1, y1, x2, y2) of the readable signal heads in this video."""
    out = {}
    for s in scene.of_type("signal"):
        if s["name"] in names:
            (x1, y1), (x2, y2) = s["points"].min(0), s["points"].max(0)
            out[s["name"]] = (int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2)))
    return out


def lamp_state(crop):
    """'red' | 'yellow' | 'green' | 'off' for one signal-head crop (BGR).

    Colour decides, and the lamp position vetoes: in daylight an amber lamp looks reddish, but it is in the
    middle of the head. On 120 hand-labelled crops (experiments/signal_methods.py): colour alone 90%,
    position alone 93%, colour + position 95%, and no method ever confused red with green."""
    colour, where = colour_state(crop), position_state(crop)
    return colour if where in (colour, "off") else where


POSITIONS = ("red", "yellow", "green")  # top, middle, bottom lamp of a vehicle head


def position_state(crop, top_k=15, margin=15):
    """Colour-blind reading: which third of the head holds the brightest saturated pixels."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2].astype(float)
    bright = []
    for rows in np.array_split(np.arange(crop.shape[0]), 3):
        lamp = v[rows][s[rows] >= MIN_SAT]
        bright.append(float(np.sort(lamp)[-top_k:].mean()) if lamp.size >= top_k else 0.0)
    order = np.argsort(bright)[::-1]
    if bright[order[0]] < MIN_SCORE:
        return "off"
    if bright[0] >= MIN_SCORE and bright[0] >= bright[2] + margin:
        return "red"  # top lamp lit, alone or with amber (the red+amber phase before green is still red)
    return POSITIONS[order[0]] if bright[order[0]] - bright[order[1]] >= margin else "off"


def colour_state(crop):
    """Colour reading: the band whose brightest saturated pixels clearly outshine the others."""
    if crop.size == 0:
        return "off"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    score, vals = {}, {}
    for colour, ranges in BANDS.items():
        band = np.zeros(h.shape, bool)
        for lo, hi in ranges:
            band |= (h >= lo) & (h <= hi)
        vals[colour] = v[band & (s >= MIN_SAT)]
        score[colour] = float(np.sort(vals[colour])[-TOP_K:].mean()) if vals[colour].size >= TOP_K else 0.0
    ranked = sorted(score, key=score.get, reverse=True)
    best, second = score[ranked[0]], score[ranked[1]]
    if best < MIN_SCORE:
        return "off"
    if best - second >= MARGIN:
        return ranked[0]
    # tie (an overexposed lamp at dusk bleeds into the neighbouring hue band): the band with clearly more
    # near-peak pixels is the lit one
    near = {c: int((vals[c] >= best - 30).sum()) for c in ranked[:2]}
    a, b = ranked[0], ranked[1]
    if near[a] >= 1.5 * near[b]:
        return a
    if near[b] >= 1.5 * near[a]:
        return b
    return "off"


def states(frame, rects):
    return {name: lamp_state(frame[y1:y2, x1:x2]) for name, (x1, y1, x2, y2) in rects.items()}


def main():
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from scene import Scene

    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--background", default="", help="median background of this video (computed if missing)")
    p.add_argument("--every", type=float, default=0.5, help="seconds between samples")
    p.add_argument("--out", default="runs/signals")
    a = p.parse_args()

    bg = cv2.imread(a.background) if a.background else None
    rects = boxes(Scene.for_video(a.video, background=bg))
    cap = cv2.VideoCapture(a.video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(a.every * fps)))
    rows = []
    for idx in range(n):
        if idx % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        rows.append({"t_sec": round(idx / fps, 2), **states(frame, rects)})
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.video).stem
    df.to_csv(out / f"{name}.csv", index=False)

    colors = {"red": "#dc2626", "yellow": "#f59e0b", "green": "#16a34a", "off": "#d4d4d4"}
    fig, ax = plt.subplots(figsize=(12, 1 + 0.6 * len(rects)))
    for i, sig in enumerate(rects):
        ax.scatter(df.t_sec, [i] * len(df), c=[colors[s] for s in df[sig]], marker="|", s=300)
    ax.set_yticks(range(len(rects)), list(rects))
    ax.set_xlabel("time (s)")
    ax.set_title(f"{name}: signal states")
    plt.tight_layout()
    plt.savefig(out / f"{name}.png", dpi=110)
    print(name, {sig: df[sig].value_counts().to_dict() for sig in rects})


if __name__ == "__main__":
    main()
