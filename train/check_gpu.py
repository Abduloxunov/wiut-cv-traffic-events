"""Check a new machine: GPU visible to PyTorch, and how fast our detector runs on a 4K frame.

  python train/check_gpu.py
Prints GPU name / VRAM and milliseconds per 4K frame for YOLO26m @1280 (Part A) and YOLO26s @960 (Part B).
"""
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def bench(weights, imgsz, frame, n=30):
    model = YOLO(str(ROOT / "weights" / weights))
    kw = {"device": 0, "half": True} if torch.cuda.is_available() else {"device": "cpu"}
    for _ in range(3):  # warm-up
        model.predict(frame, imgsz=imgsz, verbose=False, **kw)
    t = time.perf_counter()
    for _ in range(n):
        model.predict(frame, imgsz=imgsz, verbose=False, **kw)
    return (time.perf_counter() - t) / n * 1000


def main():
    print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB VRAM, CUDA {torch.version.cuda}")
    else:
        print("WARNING: no GPU visible - reinstall torch with CUDA (see setup script)")
    frame = cv2.imread(str(ROOT / "scene" / "background.jpg"))  # a real 4K frame of our camera
    for weights, imgsz in (("yolo26m.pt", 1280), ("yolo26s.pt", 960)):
        ms = bench(weights, imgsz, frame)
        # Part A runs on every 3rd frame of 29.97 fps video = 10 frames per video second
        print(f"{weights} @{imgsz}: {ms:.1f} ms/frame -> {ms * 10 / 1000:.2f} s per video second at stride 3")


if __name__ == "__main__":
    main()
