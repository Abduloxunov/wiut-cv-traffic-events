# WIUT Hackathon 2026 — CV track: traffic event detection

Detects traffic events in a fixed 4K road camera (`detect_events`, Part A) and scores accident risk causally
(`RiskEstimator`, Part B), with the starter-kit interface: `solution.py` + the unchanged `run_submission.py` and
`evaluate.py`.

## Install and run
```bash
pip install -r requirements.txt
bash weights/download.sh          # once, with internet: YOLO26m + YOLO26s (COCO-pretrained, Ultralytics)
python run_submission.py --videos /data/test --out predictions.json
```
Python 3.10+; runs offline after the weights are fetched. On a machine without a GPU the harness's 3 × duration limit
is not met (tracking a 4K video on CPU is ~10 × real time); `PART_A_BUDGET` / `RISK_BUDGET` (x duration) only exist
for local CPU experiments.

## Approach
```
video ──> YOLO26m @1280, every 3rd frame ──> ByteTrack (tuned) ──> tracks (+ speed, ground point)
   │                                                                  │
   ├─ first frame ─> SIFT + RANSAC homography to the reference view ─> hand-drawn zones in this video's pixels
   ├─ signal-head crops ─> lamp colour + position ─> traffic-light timeline
   └──────────────────────────────> event layer v2 (src/v2/, one module per class) ─> [[start, end, label]]
Part B: every 3rd frame ─> YOLO26s @960 + ByteTrack ─> pairwise closing speed / time-to-contact ─> DRAC ─> risk
```
**Learned:** only the detectors (YOLO26m / YOLO26s, COCO-pretrained, used as released — no fine-tuning, no training
data of ours). **Rule-based:** tracking (ByteTrack), zone alignment (SIFT), traffic-light reading (colour + lamp
position), every event class, and the Part B risk (a surrogate-safety measure). Rules and their thresholds are tuned on
our own labels of the sample videos (below).

### Event classes (Part A)
| Class | Method (src/v2/) | Dev F1* |
|---|---|---|
| illegal_turn | origin–destination + lane of origin: left turn to the bottom-left road from any lane but the leftmost | 1.00 |
| stop_line | vehicle arrives past the stop line while its (readable) signal is red and stops there; ends at green | 0.86 |
| stopped_vehicle | static groups of near-still vehicle boxes linked by overlap (not by track IDs); a queue = stands and drives off together with its neighbours | 0.83 |
| jaywalking | scene-level "someone on the carriageway outside a crossing" (perspective-scaled margin, riders excluded), k-of-n confirmation, gap bridging | 0.47 |
| congestion | ≥ 4 still vehicles in the junction ≥ 5 s, bridged over 10 s gaps | 0.36 |
| failure_to_yield | post-encroachment time: a moving vehicle's footprint covers a spot a walking pedestrian occupied on the crossing within 2 s | 0.26 |
| solid_line_crossing | vehicle footprint centre crosses (or straddles) the solid stretch of a lane divider | 0.14 |
| near_miss, red_light, wrong_way, accident, illegal_u_turn, road_obstacle, fire_smoke | not emitted: no reliable detector on the samples (an absent class that is predicted adds a zero to the macro average) | — |

\*Official `evaluate.py`, mean F1 over tIoU 0.3/0.5/0.7, on our labels of three sample videos (58+ events), with
settings chosen on the same labels — optimistic; leave-one-video-out numbers are lower (e.g. jaywalking 0.44,
failure_to_yield 0.22). Overall Score A on these labels: first rule set 0.138 → event layer v2 0.490.
Work log with every experiment: `docs/worklog.md`; labelling conventions and team decisions: `docs/labeling.md`,
`labels/notes.md`.

### Part B
`src/risk.py`: for each pair of road users that are apart now and closing, time to contact at constant velocity and
the deceleration rate needed to avoid the crash (DRAC, in object sizes / s²), strongest pair, smoothed; calibrated so
normal sample traffic stays below 0.5 (0 alarms in 7.4 min). The samples contain no accidents, so Part B is not
validated on real crashes.

### Datasets and licences
No training in this submission. Models: Ultralytics YOLO26 (AGPL-3.0), COCO-pretrained. Our own labels of the sample
videos are used only to choose rule thresholds; the footage itself stays within the team (AI Lab data condition).

## Reproducibility
- Seeds fixed (`SEED = 0`: Python, NumPy, PyTorch; cuDNN deterministic).
- Non-deterministic: GPU inference order and floating point can change detections slightly, and both parts stop
  processing frames when their time budget is used up (`PART_A_BUDGET`, `RISK_BUDGET`), so a much slower machine gives
  fewer frames.
- `predictions_samples.json`: our output on the four sample videos, in the harness format (validated with
  `evaluate.py --validate-only`). No GPU was available before the deadline, so it was produced on CPU by
  `tools/make_predictions_samples.py` with solution.py's exact steps: Part A = event layer on YOLO26m @1280 +
  ByteTrack tracks of every 3rd frame (the same settings as solution.py, saved by `src/detect_track.py`), lights read
  on every 3rd frame, zones aligned on the first frame; Part B = `solution.RiskEstimator` fed every frame like
  `run_submission.py` (time budget lifted on CPU). On a T4 the command above should reproduce it up to GPU
  floating-point differences.

## Layout
| Path | What |
|---|---|
| `solution.py` | Harness interface; `EVENT_LAYER=v1` switches back to the first rule set (`src/rules.py`) |
| `src/tracker.py`, `src/bytetrack_tuned.yaml` | YOLO26 + ByteTrack |
| `src/scene.py`, `src/align.py`, `scene/` | Zones (`zones.json`, drawn on `background.jpg`) aligned per video |
| `src/signals.py` | Traffic-light reader |
| `src/v2/` | Event layer v2: one module per class, `pipeline.py` runs them |
| `src/risk.py` | Part B risk |
| `experiments/` | Diagnosis, searches and scoring per class against the dev labels |
| `tools/` | Zone editor, labelling tool (+ per-video bundles for teammates, local server) |
| `train/` | Scripts prepared for later training (detector fine-tune, accident verifier); not used by this submission |
| `docs/` | Work log, research, methods, class notes, labelling guide |

## Team
| Member | Role | Links |
|---|---|---|
| Davlatyor Abduloxunov (lead) | Scene map of the junction, dev-set labelling and review, research direction, pipeline and design decisions, website | [GitHub](https://github.com/Abduloxunov) · [LinkedIn](https://www.linkedin.com/in/abduloxunovdavlatyor) · [Portfolio](https://abduloxunov.github.io/) |
| Muhammadjon Xalimov | Labelling of the sample videos (dev set) | [GitHub](https://github.com/muxammadjonx07-dev) · [LinkedIn](https://www.linkedin.com/in/%D0%BC%D1%83%D1%85%D0%B0%D0%BC%D0%BC%D0%B0%D0%B4%D0%B6%D0%BE%D0%BD-%D1%85%D0%B0%D0%BB%D0%B8%D0%BC%D0%BE%D0%B2-b85340334) |
| Bexruz Xasanov | Labelling of the sample videos (dev set) | [GitHub](https://github.com/bekhruz-khasanov) |

Developed with AI coding assistance (Claude); design decisions and labels by the team.
