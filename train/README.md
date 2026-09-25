# Training on the AI Lab GPU

What we train, on which public data, with which script. Our own sample videos (labels, detector frames)
are done later on Colab/Kaggle.

## 1. Setup (once, with internet)
```bash
git clone git@github.com:Abduloxunov/wiut-cv-traffic-events.git && cd wiut-cv-traffic-events
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r train/requirements-train.txt
bash weights/download.sh                               # yolo26m.pt, yolo26s.pt
nvidia-smi                                             # check GPU + VRAM
```

## 2. What to train
| # | Model | Public data | Script | GPU time |
|---|---|---|---|---|
| 1 | **Accident verifier**: VideoMAE (or DINOv2) features + logistic head | ACCIDENT benchmark (real + synthetic CCTV), TAD | `accident_features.py` → `train_accident_head.py` | 3–6 h |
| 2 | **Detector fine-tune**: YOLO26m @1280 | UA-DETRAC (fixed traffic cameras), later our frames | `train_detector.py` | 5–15 h |
| 3 | Optional: fire/smoke detector | D-Fire | `train_detector.py` with a D-Fire YAML | 2–3 h |

## 3. Datasets (download to `data/`)
| Dataset | What | Link |
|---|---|---|
| ACCIDENT (CVPR 2026) | 2,027 real CCTV + 2,211 synthetic accident clips, impact time + type; annotations CC BY 4.0 | https://accidentbench.github.io/ · https://www.kaggle.com/competitions/accident · paper https://arxiv.org/abs/2604.09819 |
| TAD | 344 surveillance videos (277 with accidents) | https://github.com/UnicomAI/UnicomBenchmark/tree/main/TADBench |
| CADP | 1,416 CCTV accident segments from YouTube | https://arxiv.org/abs/1809.05782 |
| UCF-Crime (RoadAccidents class) | surveillance videos with temporal labels | https://www.crcv.ucf.edu/projects/real-world/ · https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset |
| UA-DETRAC | 100 fixed-camera traffic sequences, 1.21 M vehicle boxes | https://sites.google.com/view/daweidu/projects/ua-detrac · https://huggingface.co/datasets/xujiazhen/ua_detrac · https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset |
| D-Fire | 21,527 fire/smoke images, YOLO format | https://github.com/gaia-solutions-on-demand/DFireDataset |

Kaggle downloads need an API token: `pip install kaggle`, put `kaggle.json` in `~/.kaggle/`, then
`kaggle competitions download -c accident` / `kaggle datasets download -d dtrnngc/ua-detrac-dataset`.
Record every licence in the main README before training.

## 4. Models (pretrained weights)
| Model | Link | Licence |
|---|---|---|
| YOLO26m / YOLO26s | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt · docs https://docs.ultralytics.com/models/yolo26 | AGPL-3.0 |
| VideoMAE base (Kinetics) | https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics | CC-BY-NC 4.0 (check) |
| DINOv2 base | https://huggingface.co/facebook/dinov2-base | Apache-2.0 |
| YOLOE (open vocabulary, obstacles/fire, no training) | https://docs.ultralytics.com/models/yoloe | AGPL-3.0 |

## 5. Commands
```bash
# 1. accident verifier: build data/accident_windows.csv (video,start_sec,end_sec,label) from the dataset
#    annotations: positives = impact_time-1s .. impact_time+2s, negatives = normal stretches
python train/accident_features.py --csv data/accident_windows.csv --backbone videomae --out data/feats_videomae.npz
python train/accident_features.py --csv data/accident_windows.csv --backbone dinov2   --out data/feats_dinov2.npz
python train/train_accident_head.py --feats data/feats_videomae.npz --out weights/accident_head.joblib
python train/train_accident_head.py --feats data/feats_dinov2.npz   --out weights/accident_head_dinov2.joblib
#    keep the backbone with the higher grouped ROC-AUC / AP

# 2. detector: convert UA-DETRAC to YOLO format with classes 0 person,1 bicycle,2 car,3 motorcycle,4 bus,5 truck
#    (DETRAC car->2, bus->4, van->2, others->5), write data/detrac.yaml, then
python train/train_detector.py --data data/detrac.yaml --model weights/yolo26m.pt --epochs 40 --imgsz 1280 --batch 8

# timing run on the lab GPU (competition-like hardware): put 4K samples in samples/, then
python run_submission.py --videos samples --out predictions_samples.json
```
