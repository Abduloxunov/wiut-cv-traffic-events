"""Extract signal-head crops at sampled times for hand labelling (ground truth for the signal-state methods).

Per video: 25 random times + 15 times near state changes of the current reader (the hard cases).
Writes experiments/data/signal_crops.npz and numbered contact sheets for labelling.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import signals  # noqa: E402
from scene import Scene  # noqa: E402

RUNS = Path("../runs")
OUT = Path("experiments/data")
rng = np.random.default_rng(0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops, meta = [], []
    for vid in ["C3897", "C3902", "C3905"]:
        video = f"../sample_videos/{vid}.MP4"
        rects = signals.boxes(Scene.for_video(video, background=cv2.imread(f"scene/bg_{vid}.jpg")))
        timeline = pd.read_csv(RUNS / "signals" / f"{vid}.csv")
        change = timeline.index[timeline.signal_5 != timeline.signal_5.shift()].to_numpy()[1:]
        near = timeline.t_sec.to_numpy()[np.clip(rng.choice(change, 15, replace=len(change) < 15) + rng.integers(-2, 3, 15),
                                                 0, len(timeline) - 1)]
        times = np.sort(np.r_[rng.uniform(0, timeline.t_sec.max(), 25), near])
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            for sig in ("signal_5", "signal_4"):
                x1, y1, x2, y2 = rects[sig]
                crops.append(cv2.resize(frame[y1:y2, x1:x2], (48, 96), interpolation=cv2.INTER_AREA))
                meta.append({"video": vid, "t_sec": round(float(t), 2), "signal": sig})
    np.savez_compressed(OUT / "signal_crops.npz", crops=np.stack(crops))
    pd.DataFrame(meta).to_csv(OUT / "signal_crops.csv", index_label="id")
    # contact sheets of signal_5 crops, 40 per sheet, upscaled, numbered by id
    ids = [i for i, m in enumerate(meta) if m["signal"] == "signal_5"]
    for k in range(0, len(ids), 40):
        tiles = []
        for i in ids[k:k + 40]:
            tile = cv2.resize(crops[i], (96, 192), interpolation=cv2.INTER_NEAREST)
            tile = cv2.copyMakeBorder(tile, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(tile, str(i), (4, 16), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % 10:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[r:r + 10]) for r in range(0, len(tiles), 10)])
        cv2.imwrite(str(OUT / f"sheet_signal5_{k // 40}.jpg"), sheet)
    print(len(meta), "crops")


if __name__ == "__main__":
    main()
