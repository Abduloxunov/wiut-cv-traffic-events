# WIUT Hackathon 2026 — CV track: traffic event detection

Work in progress for the elimination round. Detects traffic events in a fixed road camera
(`detect_events`) and scores accident risk causally (`RiskEstimator`), per the starter-kit interface.

## Layout
| Path | What |
|---|---|
| `solution.py` | The interface the harness imports (starter kit; being implemented) |
| `run_submission.py`, `evaluate.py`, `examples/` | Starter kit, unchanged |
| `src/detect_track.py` | YOLO26 + ByteTrack over a video → `runs/<video>/tracks.csv` + annotated video |
| `src/scene_stats.py` | EDA from tracks: road/person heatmaps, learned lane directions, counts over time |
| `tools/zone_editor.html` | Draw the scene layout (road, crossings, stop lines, signals…) → `scene/zones.json` |
| `tools/label_tool.html` | Label events on a video timeline → `labels.json` in the exact ground-truth format |
| `src/align.py` | Maps the reference view to each video (SIFT + homography); framing shifts between recordings |
| `src/check_zones.py` | Renders the zones and checks them against tracked traffic |
| `src/propose_events.py` | Rule-based event proposals (baseline v0) from tracks + zones |
| `src/bench_detectors.py` | Label-free comparison of detector / tracker configs |
| `docs/` | Work log, methods comparison, research notes, labelling guide |
| `scene/background.jpg` | Reference background (median of sample C3897) that zones are drawn on |
| `weights/download.sh` | Fetches model weights |
| `samples/` | Put the sample `.MP4` files here (not committed) |

## Setup
```bash
pip install -r requirements.txt
bash weights/download.sh
```

## Zone editor
```bash
python -m http.server 8765
# open http://localhost:8765/tools/zone_editor.html, draw, then "Download zones.json" into scene/
```

## Notes
- The sample videos are 4K (3840×2160) at 29.97 fps. Camera framing shifts slightly between recordings,
  so zones drawn on the reference background are aligned to each video by SIFT feature matching + homography.
