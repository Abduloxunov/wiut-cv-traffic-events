"""Extract clip features for the accident verifier from any CCTV accident dataset (ACCIDENT, TAD, ...).

Input: a CSV with one row per clip window:
  video,start_sec,end_sec,label        (label 1 = contains the collision, 0 = normal traffic)
Make positives as a 2-4 s window around each annotated impact time, negatives from normal parts of the same
datasets and from our own sample videos (hard negatives for this camera).

Output: an .npz with one feature vector per row (backbone = VideoMAE on 16 frames, or DINOv2 mean-pooled).

Example (lab GPU):
  python train/accident_features.py --csv data/accident_windows.csv --backbone videomae --out data/feats_videomae.npz
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

BACKBONES = {
    "videomae": "MCG-NJU/videomae-base-finetuned-kinetics",
    "dinov2": "facebook/dinov2-base",
}


def read_window(path, start, end, n=16, size=224):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    for t in np.linspace(start, end, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            f = frames[-1] if frames else np.zeros((size, size, 3), np.uint8)
        frames.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--backbone", choices=list(BACKBONES), default="videomae")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    from transformers import AutoImageProcessor, AutoModel, VideoMAEImageProcessor, VideoMAEModel
    name = BACKBONES[a.backbone]
    if a.backbone == "videomae":
        proc, model = VideoMAEImageProcessor.from_pretrained(name), VideoMAEModel.from_pretrained(name)
    else:
        proc, model = AutoImageProcessor.from_pretrained(name), AutoModel.from_pretrained(name)
    model = model.to(a.device).eval().half() if a.device == "cuda" else model.eval()

    rows = pd.read_csv(a.csv)
    feats = []
    with torch.no_grad():
        for i, r in rows.iterrows():
            frames = read_window(r.video, r.start_sec, r.end_sec)
            if a.backbone == "videomae":
                x = proc(frames, return_tensors="pt")["pixel_values"].to(a.device)
                h = model(pixel_values=x.half() if a.device == "cuda" else x).last_hidden_state.mean(1)
            else:
                x = proc(images=frames, return_tensors="pt")["pixel_values"].to(a.device)
                h = model(pixel_values=x.half() if a.device == "cuda" else x).pooler_output.mean(0, keepdim=True)
            feats.append(h.float().cpu().numpy()[0])
            if i % 100 == 0:
                print(f"{i}/{len(rows)}")
    np.savez_compressed(a.out, X=np.stack(feats), y=rows.label.to_numpy(), video=rows.video.to_numpy())
    print("saved", a.out, np.stack(feats).shape)


if __name__ == "__main__":
    main()
