# Work log

Chronological record of what we tried, what we measured and what we learned. Source material for the
technical report and the website. Newest entries at the bottom.

## 24 Sep 2026

**Task analysis** (task PDF + starter kit)
- Final score = 0.6·Model + 0.25·Website + 0.15·Code; Model = 0.7·Part A + 0.3·Part B.
- Part A is macro F1 over classes in the ground truth *or in our predictions*, averaged over tIoU 0.3 / 0.5 / 0.7.
  A class we predict that is absent from the test set adds a zero to the average, so rare classes need
  high confidence before we emit them.
- The package failing to run gives Model = 0 (60% of the total). Clean-machine testing is top priority.

**Environment**
- Dev laptop: 12 CPU threads, no GPU. Python 3.12 venv, torch 2.14 (CPU), ultralytics 8.4.161.
- Evaluation machine: T4 16 GB, 8 CPU cores, offline, time budget 3× video duration for Part A + B.

**First detection + tracking run** (`src/detect_track.py`, YOLO26s, imgsz 1280, every 3rd frame, ByteTrack)
- C3905 (127.6 s): 819 s on CPU = 6.4× real time. 241 car tracks, 429 person tracks.
- C3902 (317.8 s): 2208 s on CPU = 6.95× real time. 550 car tracks, 983 person tracks.
- Pretrained COCO detector finds cars, buses, trucks, cyclists and small far pedestrians at dusk without training.
- ID churn: person tracks 107 → 91 in 20 s after tuning ByteTrack (`src/bytetrack_tuned.yaml`:
  track_high 0.3, new_track 0.4, buffer 60, conf 0.1 so the low-score stage is used).
- Class flicker per track (car ↔ truck); fix = majority class per track (done in later scripts).

**EDA findings** (`src/scene_stats.py`)
- Signalised intersection with 3 zebra crossings; vehicle signal heads visible and red lamps readable from pixels.
- Learned lane directions from track motion: far carriageway right → left, near carriageway top-left → junction.
- Car count per frame rises and falls with the signal cycle (13–46 cars, peaks ~80–160 s apart in C3902).
  → `congestion` must not fire on a normal red queue.
- End of C3905 shows the whole junction jammed: a real `congestion` example.
- Most pedestrians are on the far pavement / bus stop; few on crossings.

**Decode speed**
- 4K H.264 decode with OpenCV `grab()`: ~36 fps on 12 threads. The harness decodes every frame again for
  Part B, so decode alone may take ~1× duration on the 8-core evaluation machine. Runtime is a design constraint.

## 25 Sep 2026

**Starter kit** (`run_submission.py`, `evaluate.py`)
- Harness order: `detect_events()` first, then it re-opens the video and streams every frame into
  `RiskEstimator.step()`. One shared budget; Part A overrun → Part B skipped → video scored empty.
- If the test set has no accidents, Model = Score A (Part B ignored).
- Alarms starting inside an accident are dropped, not counted as false alarms.
- Example labels use red_light, accident, jaywalking, near_miss, wrong_way, stopped_vehicle, 2–15 s long.
- `camera.md` does not exist; organizers said to disregard it. We draw the scene layout ourselves.

**Sample videos**: C3897 (daylight, 317.8 s), C3902 (dusk, 317.8 s), C3905 (dusk, 127.6 s); all 3840×2160 @ 29.97 fps.
Camera splits recordings into fixed-size chunks (C3897 and C3902 have identical byte sizes, different content).

**Camera framing is not identical between recordings**
- Median backgrounds show ~125 px shift in 4K plus slight zoom/rotation between C3897 and the dusk videos.
- Fix: draw zones once on a reference background (median of C3897) and map them to each video with a
  homography from SIFT matches between backgrounds (`src/align.py`). 138–144 RANSAC inliers per video;
  the overlay shows no double lines.

**Scene zones**
- Built `tools/zone_editor.html` (polygons, lines, arrows, rects; exports `scene/zones.json`).
- Team drew 57 shapes. `src/check_zones.py` checks them against 7 min of tracked traffic:
  - main-avenue direction arrows agree with real traffic (0–9° off, thousands of observations each);
  - `lane_dir_8` was reversed (fixed); a duplicate name (fixed); bus stop drawn on the shelter, added `bus_bay` lane;
  - arrows inside the junction disagree with traffic — expected, cars turn every way there; direction checks are
    switched off inside `intersection_1`;
  - 9% of vehicle points fall off the drawn road (mostly parked cars and a side street — fine);
  - 12% of person points are on the road outside crossings, mostly right at crossing / bus-bay / island edges
    → rules need a margin and must ignore people inside vehicle boxes (passengers, riders).

**Research** (see `docs/research.md`)
- Classic design confirmed by public repos and AI City Challenge winners: detector + tracker + rules on zones,
  background modelling for stalled vehicles, HSV thresholds for signal state.
- A repo (`erkinovvv/roadwatch`) listed almost exactly our class list (likely another hackathon team); it was
  deleted (404) before we could read it. We do not use other teams' code.

**Detector / tracker benchmark** (`src/bench_detectors.py`, C3897 60–80 s, every 3rd frame, 1920-wide input,
agreement measured against YOLO26m as pseudo ground truth) — results in `docs/methods.md`.

**Rule proposals v0** (`src/propose_events.py`) on C3905: 37 proposals
(12 jaywalking, 22 failure_to_yield, 1 congestion 85–112 s, 2 wrong_way, 0 stopped_vehicle).
Visual review of the middle frame of each proposal:
- congestion 85–112 s matches the jam seen at the end of the video;
- some jaywalking proposals are real (people walking through the junction between the islands);
- many failure_to_yield proposals are false: a pedestrian waiting at the *end* of a long crossing makes every
  car passing the *other end* a violation. Fix: pedestrian must be inside the crossing core and near the car's path;
- people standing at kerbs and the bottom edge of the frame create jaywalking false positives.
Conclusion: proposals are useful as pre-labels, not as final answers; real dev labels are needed to tune.

**Labelling tool** (`tools/label_tool.html`): video + per-class timeline, keyboard marking, proposals shown dashed
(accept / fix / delete), exports `evaluate.py` ground-truth format. Tested in the browser with a clip and the
C3905 proposals. Bug found in testing: class keys relabelled the selected segment silently → now class keys only set
the class for new segments; Shift+key relabels.

**Proxies for labelling**: 1280-wide H.264 copies of the samples (FFmpeg from the `imageio-ffmpeg` pip package;
OpenCV on Windows has no H.264 encoder). Frame times are identical to the originals, so labels transfer 1:1.
