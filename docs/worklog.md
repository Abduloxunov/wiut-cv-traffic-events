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

**End-to-end solution v1** (afternoon)
- Code split into modules shared by the harness, the CLIs and (later) the web demo:
  `src/tracker.py` (detect + track, one pass, early stop on a deadline), `src/scene.py` (zones aligned per video,
  masks), `src/rules.py` (event rules + segment merging), `src/align.py` (homography, median background).
- `solution.py` now implements Part A: YOLO26m @1280, every 3rd frame, ByteTrack → zones aligned with a median
  background built from frames the tracker already decoded (no second decode) → rules for stopped_vehicle,
  jaywalking, failure_to_yield, congestion, wrong_way. Fixed seeds. Part B still returns 0.
- Ran the unchanged official `run_submission.py` on a 20 s 4K clip cut with `-c copy` (same codec): valid output
  (`evaluate.py --validate-only` passes).
- Runtime measured through the harness on the laptop CPU: the harness's own Part B decode of 4K took 21.8 s for a
  21 s clip (~1.04× real time on 12 threads; expect ~1.3× on the 8-core evaluator). So Part A tracking is capped at
  1.3× duration (`PART_A_BUDGET`, env override for local CPU experiments). The cap worked: the CPU run stopped early
  and still returned a valid result inside the budget.
- With the cap lifted (CPU, 240 s for 21 s), the clip from the C3905 jam gives: congestion 1.0–18.9 s (the real jam),
  one jaywalking segment, four failure_to_yield segments.
- failure_to_yield rule v1: pedestrian must be on the same crossing (15 px margin) AND within 4 vehicle-box-heights of
  the vehicle; people inside vehicle boxes ignored. Jaywalking v1 ignores people at the frame border.
  Proposal counts v0 → v1: failure_to_yield 42→29 (C3897), 37→30 (C3902), 22→18 (C3905); jaywalking 21→17, 23→22, 12→7.
- C3897 tracking run was interrupted at 257/318 s (app closed); its proposals cover 0–257 s.

**Traffic-light reader: 5 methods tested** (`experiments/signal_methods.py`, 120 hand-labelled crops of the vehicle
head signal_5, 40 per video, half at random times and half near state changes)
| method | accuracy | C3897 day | C3902 dusk | C3905 dusk | red↔green errors |
|---|---|---|---|---|---|
| fixed HSV thresholds | 0.600 | 0.100 | 0.725 | 0.975 | 0 |
| relative colour (brightest saturated pixels per band) | 0.900 | 0.875 | 0.900 | 0.925 | 0 |
| lamp position (brightest third of the head) | 0.925 | 1.000 | 0.825 | 0.950 | 0 |
| **colour + position veto (production)** | **0.950** | **1.000** | **0.875** | **0.975** | **0** |
| 1-NN on HSV thumbnails, leave-one-video-out | 0.625 | 0.125 | 0.875 | 0.875 | 0 |
- Daylight amber looks reddish to a colour rule; the lamp position fixes it. A learned model fails on daylight
  because only one daylight video exists. No method ever confused red with green.
- Only 2 of 7 drawn heads face the camera (signal_4 pedestrian, signal_5 vehicle); gantry heads show their backs.
- signal_5 governs stop_line_1: 343 of 351 stop-line crossings happened on green. Cycle ≈ 75–80 s (red ≈ 40 s,
  green ≈ 35 s, flashing green, yellow); red+amber is shown before green (counted as red).
- red_light review: 6 raw candidates → 3 were drivers moving off < 1 s before green (not violations) → rule now
  requires red 0.5 s before and ≥ 1 s after the crossing; 2 genuine late runners remain (bus at 57.6 s and car at
  132.5 s of C3897, both 2.5 s into red).

**Part B risk estimator** (`src/risk.py`, replayed on saved tracks by `experiments/risk_on_tracks.py`; the samples
have no accidents, so every alarm is false — the one number we can measure)
| version | change | false alarms / hour | median score |
|---|---|---|---|
| v1 | time to closest approach between box centres | 89 | 0.80 |
| v2 | ground points; ignore pairs already overlapping in the image | 138 | 0.69 |
| v3 | per-pair persistence, true time-to-contact | 97 | 0.71 |
| v4 | contact distance 0.35 / 0.25 | 162 / 283 | 0.64 / 0.53 |
| v5 | ignore pairs on opposite one-way carriageways | 162 | 0.64 |
| v6 | tracks ≥ 8 observations, ignore tiny far boxes | 178 | 0.63 |
| **v7** | **severity = DRAC (closing² / 2·gap), 0.5 at DRAC 20 sizes/s² (above the 99.97th pct of normal traffic)** | **0** | **0.07** |
- Visual check of the worst v6 false alarm: two far cars in one lane, one half hidden behind a truck; occlusion
  shrinks and shifts its box, which looks like fast closing. Image-plane TTC is inherently noisy on this oblique view.
- v7 keeps a continuous, ranked score (AP) and stays below 0.5 on normal traffic (max 0.44). Untested on real
  crashes (none in the samples); the next improvement is a bird's-eye-view (metric) ground plane.
- Part B cannot lower the score (all its terms are ≥ 0); the only risk is runtime, so RiskEstimator uses YOLO26s at
  960 px every 3rd frame and stops inferring if it exceeds 0.25× the video length.

## 27 Sep 2026
**To do after labelling: redraw the solid lines in `scene/zones.json`.** `solid_line_1` … `solid_line_5` were drawn
to the edge of the frame, but only the last stretch before the stop lines (about 5–6 m) is actually solid; the rest
is dashed. Shorten each to the solid part (check on a daylight frame of C3897), otherwise `solid_line_crossing`
would flag every normal lane change upstream.

**Rule fix found while labelling: `stop_line` false positive.** `rules.stop_line` flags any vehicle standing still
past the line while the signal is red, so a car that crossed on green and is held past the line by a queue of
turning cars gets flagged once the light turns red. Fix: require the vehicle to *arrive/stop* past the line while
red (its crossing of the line, or its first still frame there, happens on red), not just to be standing there
during red. Check against the labels afterwards.

**Rule fix found while labelling: `congestion` fires on partial-lane queues.** The rule counts ≥ 4 still vehicles
in the junction for ≥ 5 s, so one backed-up lane (often a turn lane) triggers it while the other lanes of that
direction flow. The definition needs *all lanes of a direction* stopped or crawling. Fix: per direction (lane_dir
zones), require still/crawling vehicles across every lane of that direction, and ignore queues that clear on green.
