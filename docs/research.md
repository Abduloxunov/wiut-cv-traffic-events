# Research notes

What we looked at, what we took from it. Accessed 24–25 Sep 2026.

## Similar systems (detector + tracker + rules)
| Source | What it does | Takeaway for us |
|---|---|---|
| [ilhamfzri/traffic-violation-detection](https://github.com/ilhamfzri/traffic-violation-detection) | YOLOv5 + SORT; red light, helmet, wrong way | Same rule-on-tracks design |
| [BYTEZEN-11/TRAFFIC](https://github.com/BYTEZEN-11/TRAFFIC) | YOLO + Kalman; rule engine for red light, speeding, wrong way; signal state by HSV | HSV on a fixed ROI is the standard cheap signal-state method |
| [luthfirahmn09/smart-traffic-vision](https://github.com/luthfirahmn09/smart-traffic-vision) | YOLOv8 + ByteTrack; congestion (density in ROI), accidents as "sudden stop + box overlap" candidates, rules on per-camera JSON zones | Accident heuristic = candidate generator, needs verification |
| [Sayali2005/Traffic-signal-violation-detection](https://github.com/Sayali2005/Traffic-signal-violation-detection) | YOLOv8 + tracking; red light = stop-line crossing on red | Same red-light rule |
| [Wrong-way detection with YOLO + centroid tracking (arXiv 2210.10226)](https://arxiv.org/pdf/2210.10226) | Direction vs expected lane direction | Same as our lane-arrow check |
| `erkinovvv/roadwatch` | Listed our exact class list (YOLO11n + ByteTrack). Deleted (404) on 25 Sep | Likely another team; not used |

## Stalled vehicles and crashes (AI City Challenge, traffic anomaly track)
- 2021 winner, [Good Practices and A Strong Baseline for Traffic Anomaly Detection (arXiv 2105.03827)](https://ar5iv.labs.arxiv.org/html/2105.03827),
  code [Endeavour10020/AICity2021-Anomaly-Detection](https://github.com/Endeavour10020/AICity2021-Anomaly-Detection):
  background modelling (MOG) + detection on the background finds vehicles that stay still; start time refined by
  backtracking with SSIM. F1 0.95 on the 2021 test.
- 2019 winner [ShuaiBai623/AI-City-Anomaly-Detection](https://github.com/ShuaiBai623/AI-City-Anomaly-Detection):
  background modelling + perspective map + spatio-temporal matrix.
- Takeaway: for `stopped_vehicle` and `road_obstacle`, detecting on the *background* (things that stay still) is more
  robust than tracking, which breaks IDs over long stops.

## Signal state
- HSV thresholding inside a fixed ROI; red needs two hue bands (0–10 and 170–179); pick the colour with the most
  bright pixels ([summary of common practice](https://github.com/muaz8172/traffic_light2),
  [OpenCV example](https://github.com/RomeroRodriguezD/Traffic-Lights-Tracking-and-Color-Detection-OpenCV)).

## Surrogate safety measures (Part B, near_miss)
- Time-to-collision (TTC) and post-encroachment time (PET) are the standard video-based conflict measures
  ([Kittelson overview](https://www.kittelson.com/ideas/breaking-down-video-based-conflict-monitoring/),
  [UF video-based traffic ML book](https://www.cise.ufl.edu/~tmishra/trafficml/book.html)).

## Temporal annotation tools
- [VIA video annotator](https://www.robots.ox.ac.uk/~vgg/software/via/app/via_video_annotator.html): browser-only,
  temporal segments; general-purpose export format.
- Label Studio: timeline segmentation is limited ([issue #3693](https://github.com/HumanSignal/label-studio/issues/3693)).
- [TemporalEventAnnotator](https://github.com/HaydenFaulkner/TemporalEventAnnotator): Qt desktop app.
- We built `tools/label_tool.html` instead: exports the exact `evaluate.py` format, shows our rule proposals as
  pre-labels, per-class timeline, class definitions on screen.

## Datasets and models (from earlier research)
- CCTV accident data: ACCIDENT benchmark (2,027 real + 2,211 synthetic clips, annotations CC BY 4.0), TAD (344 videos),
  CADP, SO-TAD, TU-DAT. Dashcam sets (DoTA, CCD, DAD, BDD100K) are the wrong viewpoint.
- Detectors: YOLO26 (AGPL-3.0), RF-DETR (Apache-2.0). Fire/smoke: D-Fire.

## Accidents from fixed CCTV: ACCIDENT @ CVPR 2026 (zero-shot, real CCTV test set) — researched 25 Sep
- [ACCIDENT benchmark paper](https://arxiv.org/abs/2604.09819): 2,027 real + 2,211 synthetic clips; tasks = impact
  time, impact location, collision type; best baselines ~0.41 harmonic mean.
- [Zero-shot VLM + tracking pipeline](https://arxiv.org/abs/2608.08867): Qwen3-VL-32B coarse-to-fine + YOLO11x + BoT-SORT,
  0.504 (≈ +22% over the best baseline). Too large for our 5 GB / T4 budget; the idea (cheap candidates, expensive
  verification only on windows) is what we reuse.
- [SynCrash](https://arxiv.org/abs/2608.29759): VideoMAEv2-giant trained on CARLA synthetic clips, 0.40 (17th).
  Key ablation: **physics heuristics (box overlap, trajectory intersection, approach velocity) beat complex relational
  models on noisy CCTV**; object detection beat attention maps for localisation.
- [Modular zero-shot pipeline](https://arxiv.org/abs/2604.09685) + [repo](https://github.com/Amey-Thakur/ACCIDENT-CVPR-2026)
  (concepts only): impact time = strongest z-score peak (τ = 1.5) of mean absolute frame differences smoothed over 5
  frames; location = weighted centroid of Farnebäck optical flow; type = CLIP prompts. Score 0.25.
- [Two-pass VLM grounding](https://arxiv.org/abs/2605.01512): sparse-frame VLM pass for a coarse time, then tracking
  in a ±2 s window to refine.

## Collision anticipation
- Nexar dashcam challenge (Kaggle 2025): top solutions = VideoMAEv2 clip classifiers
  ([2nd place write-up](https://www.kaggle.com/competitions/nexar-collision-prediction/writeups/siuuuuuuu-2nd-place-solution-public-private-lb-vid)).
  Dashcam viewpoint, needs thousands of labelled clips: not transferable to our fixed camera without data.
- CCTV forecasting ([CADP](https://ar5iv.labs.arxiv.org/html/1809.05782)): DSA-LSTM warns ~1.4 s early at 80% recall.
- Surrogate safety measures: TTC < 1.5 s serious conflict, 1.5–3 s slight
  ([CCTV conflict analysis](https://www.researchsquare.com/article/rs-7894809/v1)); TTC false-alarms at urban
  intersections where turning/yielding/braking break its constant-velocity assumption; practitioners rectify to a
  bird's-eye view first. We use DRAC (deceleration rate to avoid crash) as the severity, which proved far more stable.

## Open-vocabulary detection (obstacles, fire/smoke without training)
- [YOLO-World](https://docs.ultralytics.com/models/yolo-world), [YOLOE-26](https://pyimagesearch.com/2026/08/24/yolo26-open-vocabulary-object-detection-with-yoloe-26/):
  text-prompted detection ("debris", "tire", "cardboard box", "dog", "smoke", "fire") in the Ultralytics stack.

## GitHub search (via `gh`)
- Accident-detection repos for CCTV are mostly image classifiers or YOLO "accident" classes; the anticipation repos
  (CCD/UString, DSTA, DRIVE) are dashcam-based. Nothing ready-made fits a fixed camera + temporal segments + our metric,
  which confirms building our own rules on tracks.
