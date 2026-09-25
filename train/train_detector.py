"""Fine-tune the YOLO26 detector on a YOLO-format dataset (UA-DETRAC converted, or frames from our videos).

The dataset YAML must use our six COCO-named classes in this order so the fine-tuned model stays a drop-in
replacement for weights/yolo26m.pt:  0 person, 1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 truck

Example (lab GPU):
  python train/train_detector.py --data data/detrac.yaml --model weights/yolo26m.pt --epochs 40 --imgsz 1280 --batch 8
"""
import argparse
import random

import numpy as np
import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dataset YAML (train/val image dirs, names)")
    p.add_argument("--model", default="weights/yolo26m.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=1280, help="keep 1280: smaller loses distant pedestrians")
    p.add_argument("--batch", type=int, default=8, help="8 fits 16 GB at 1280 for the m model")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="yolo26m_finetune")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, seed=a.seed,
                deterministic=True, project="runs/train", name=a.name, patience=10, cos_lr=True,
                close_mosaic=5, workers=8, cache=False)
    metrics = model.val(data=a.data, imgsz=a.imgsz, device=a.device)
    print("val mAP50-95:", metrics.box.map, " mAP50:", metrics.box.map50)
    print("best weights: runs/train/%s/weights/best.pt  -> copy to weights/ to use in solution.py" % a.name)


if __name__ == "__main__":
    main()
