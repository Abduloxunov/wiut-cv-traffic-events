# Methods: options, evidence, decisions

Each pipeline stage lists the options we considered, how we evaluate them, what we measured so far, and the
current choice. "Pending" means the evidence needs our dev labels or a T4 run.

Pipeline: **decode → detect → track → align zones → per-class rules → accident/near-miss model → segments → Part B risk**

## 1. Decoding (runtime-critical)
| Option | Evidence | Status |
|---|---|---|
| OpenCV full 4K decode, use every Nth frame | 36 fps on 12 threads (laptop); the harness already spends ~1× duration decoding for Part B | Baseline |
| FFmpeg decode with in-decoder downscale (pipe raw 1280-wide frames) | FFmpeg (imageio-ffmpeg) re-encoded 20 s of 4K in 23 s on a busy CPU | To benchmark on 8 cores |
| NVDEC on the T4 (PyNvVideoCodec / torchcodec CUDA) | Frees the CPU completely | To try on Colab/Kaggle T4 |

Evaluation: wall-clock seconds per video minute on a T4 box, Part A + harness Part B together, must stay < 3× with margin.

## 2. Detector
Benchmark: C3897 60–80 s, every 3rd frame (200 frames), input 1920 wide, CPU. "Agreement" = F1 of boxes against
YOLO26m (IoU ≥ 0.5), a label-free proxy for recall/precision.

| Config | CPU fps | persons/frame | cars/frame | agree veh | agree person | tracks | median track len | fragments (<5 obs) |
|---|---|---|---|---|---|---|---|---|
| YOLO26m @1280 + ByteTrack (reference) | 0.82 | 26.2 | 19.6 | 1.00 | 1.00 | 110 | 69 | 5.5% |
| YOLO26s @1280 + ByteTrack | 1.61 | 21.4 | 19.6 | 0.95 | 0.87 | 101 | 55 | 7.9% |
| YOLO26s @1280 + BoT-SORT | 1.08 | 23.8 | 19.9 | 0.94 | 0.91 | 123 | 38 | 16.3% |
| YOLO26s @960 + ByteTrack | pending | | | | | | | |
| YOLO26n @1280 + ByteTrack | pending | | | | | | | |

Reading so far:
- Vehicles are easy: the small model agrees 95% with the medium one. People are the hard part (small, far):
  YOLO26m finds ~22% more people per frame than YOLO26s.
- On a T4 the medium model is cheap (YOLO26 docs: 1.7–11.8 ms TensorRT across sizes), so **YOLO26m is the likely
  submission detector**; YOLO26s/n for the CPU web demo.
- Other candidates not yet run: RF-DETR (Apache-2.0, strong on small objects), tiling the far half of the frame for
  pedestrians (SAHI-style).

## 3. Tracker
- **ByteTrack (tuned)** beats BoT-SORT here: half the fragments (7.9% vs 16.3%), longer tracks (55 vs 38 median),
  and faster (BoT-SORT's camera-motion compensation is wasted on a fixed camera). **Choice: ByteTrack.**
- Long stops break IDs in any tracker → `stopped_vehicle` should also use background modelling (see §5).

## 4. Scene layout
- Hand-drawn zones on a reference background, mapped per video by SIFT + homography (`src/align.py`). 138–144 inliers.
- Checked against real traffic with `src/check_zones.py`. **Done.** Test videos will be aligned the same way.

## 5. Event rules — options per class
| Class | v0 rule (implemented in `src/propose_events.py`) | Known failure modes | Next improvement |
|---|---|---|---|
| jaywalking | person's feet on road, outside crossings/pavements/islands (+45 px margin), not inside a vehicle box, ≥ 1 s | people at kerbs, frame edges, bus bay | require distance into the road; tune margin on dev labels |
| failure_to_yield | moving vehicle inside a crossing while any person is on that crossing | ped at the far end of a long crossing triggers it | person must be in the crossing core and within N m of the car's path |
| stopped_vehicle | vehicle on road, not bus bay/parking, speed < 0.12 box-heights/s for ≥ 10 s, not in a queue (< 4 others still) | ID switches during long stops | background-model stationary objects (AI City 2021 winner approach) |
| congestion | ≥ 4 vehicles still inside the junction for ≥ 5 s | ambiguous definition; red queues upstream | add per-approach "all lanes still > one signal cycle" |
| wrong_way | outside junction, moving, > 135° against nearest lane arrow for ≥ 1 s | tracker jitter at low speed | require displacement of ≥ 1 car length against the arrow |
| red_light / stop_line | not yet | needs signal state | HSV state from the drawn signal boxes, per frame, + stop-line crossing |
| solid_line_crossing | not yet | box bottom ≠ wheel position | only for clear lateral crossings of the drawn solid lines |
| illegal_turn / illegal_u_turn | not yet | no banned movements known yet | entry/exit approach per track vs a banned list |
| accident / near_miss | not yet | rare, no samples seen yet | see §6 |
| road_obstacle / fire_smoke | not planned | high false-positive risk at dusk (lights) | only with strong evidence; otherwise never emit |

Evaluation for every rule: `evaluate.py --gt dev_labels.json` per class at tIoU 0.3/0.5/0.7, plus the per-video table.
Threshold sweeps (margin, min duration, speed) run on the cached tracks in seconds.

## 6. Accident / near-miss
| Option | Cost | Status |
|---|---|---|
| Track heuristic: two boxes overlap, then both decelerate sharply and stay still | none | to implement as candidate generator |
| Off-the-shelf accident YOLO (HF `Enos-123/traffic-accident-detection-yolo11x`, MIT) on candidate frames | small | to test on our footage |
| Clip classifier on ACCIDENT/TAD (frozen DINOv2 / VideoMAE features + small head) | 1–2 GPU h on Kaggle | if time |
| Small open VLM (Qwen3-VL-2B) as verifier on candidates | ~4 GB weights, slow | only if time and budget allow |

## 7. Segments post-processing
Merge same-class segments closer than 1 s (ground truth merges simultaneous events), drop blips, optionally shift
boundaries by a learned offset. Tune on dev labels at tIoU 0.7.

## 8. Part B risk
Causal time-to-collision between tracked vehicle pairs, mapped to [0, 1] so that 0.5 ≈ "collision within 5 s";
must stay low almost always (every alarm outside the 10 s window before an accident is a false alarm).
`step()` must be light: the harness already decodes every 4K frame.

## How to choose
1. Label the 3 samples (tools/label_tool.html, start from proposals).
2. Run each option above on cached tracks → `evaluate.py` per class → keep what raises Score A.
3. Measure runtime of the chosen stack on a T4 (Kaggle/Colab) → pick detector size / stride that fits 3× with margin.
