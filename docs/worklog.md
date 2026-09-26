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

**Labelling tool made shareable** (26 Sep): fixes (Ctrl+Z undo, buttons no longer keep keyboard focus, N/P step
through the list, suggestions imported once), `tools/serve.py` (local server with Range support so videos seek),
`tools/make_label_bundles.py` (one zip per video: 1280-wide proxy + tool with that video's suggestions built in +
START.bat; nothing to install), `?store=` for separate review workspaces. Fourth sample video **C3896** (5:40) found:
proxy made, tracked (70 min on CPU), 39 suggestions, bundle built.

**Dev labels done** (27 Sep): C3897 (lead), C3902 and C3905 (first pass by teammates, then reviewed by the lead:
7 → 25 and 6 → 12 events, so first passes caught only a third to a half). 58 events, 8 classes: jaywalking 26,
solid_line_crossing 9, failure_to_yield 7, congestion 6, stopped_vehicle 4, near_miss 2, illegal_turn 2, stop_line 2.
No accidents (Part B cannot be scored on our data). Overlapping same-class segments in C3897 were merged (24 → 21).
Team decisions on ambiguous cases are in `labels/notes.md`.

**Baseline on the dev labels: Score A = 0.094** (official evaluate.py; rules v1 re-run on saved tracks). Per class:
stop_line 0.50, jaywalking 0.23, congestion 0.19, failure_to_yield 0.02 (85 predicted vs 7 real), stopped_vehicle 0,
no detector for solid_line / near_miss / illegal_turn, red_light and wrong_way only false positives. What-ifs: dropping
red_light + wrong_way → 0.118; merging jaywalking fragments ≤ 4–8 s apart → 0.133–0.138.

**Research round 2: how others build these systems** → `../reports/CCTV traffic event detection systems.md`
(notes in `../research_notes/`). 36+ repos, AI City / ACCIDENT@CVPR2026 winners, papers, vendors, Uzbekistan
context. Key points: everyone uses detector + tracker + drawn zones + light reader + one rule per class; winners win on
segment post-processing (persistence, gap bridging, one segment per event, backtracked starts); per-class fixes for our
weak classes; two other WIUT teams publicly switch off the hard classes (concepts only, nothing copied).

**Decision: rewrite the event layer from scratch, one class at a time** (keep detection, tracking, alignment, zones).
New code lives in `src/v2/`; each class is scored alone against the dev labels with leave-one-video-out tuning.

**jaywalking v2** (`src/v2/jaywalking.py`, `src/v2/segments.py`, `experiments/jaywalking/`):
- Scene-level signal "anyone on the carriageway outside a crossing"; margin to walk areas scales with the person's
  height (0.25 × box height); riders and border-cut boxes excluded; per person ≥ 40 % of samples in a 1 s window,
  runs ≥ 1 s; scene runs bridged over gaps ≤ 4 s; events < 3 s dropped.
- Result (jaywalking F1, mean of tIoU 0.3/0.5/0.7): **old 0.23 → v2 0.44–0.46 leave-one-video-out, 0.47 on all 3**
  (F1 0.58 / 0.50 / 0.33; 14 of 26 events found at 0.3). Chosen settings identical across folds.
- Tried and left switched off (mixed results): distance to any kerb (`KERB`), ignoring boxes much shorter than a
  whole person at that row (`PARTIAL`, occlusion).
- Error causes found (figures in `../figures/06_jaywalking_v2/`): people between the triangle islands and walking
  along the kerb in C3897 1:15–4:06 are unlabelled (label question); the road zone covers part of the grass strip in
  the top-right (zone error); people waiting at kerbs / planter wall; one tracker ID glued across several people;
  short 2–3 s crossings removed by the 3 s minimum; detector/track misses inside some labelled events
  (candidate present in 0–18 % of their frames).

**Figures** are now collected and grouped in `../figures/` with `INDEX.md` (local only, from footage).

### Next actions (running list)
1. Lead re-checks C3897 1:15–4:06 for jaywalking (island-to-island walkers, kerb walkers) → retune jaywalking v2.
2. Zones: remove the grass strip from the road polygon (top-right); shorten solid_line_1…5 to the real solid part.
3. stopped_vehicle v2 (next class): box-overlap persistence instead of track IDs, "queued = moves on its green".
4. Then failure_to_yield v2, congestion v2, stop_line fix, solid_line_crossing, emission gate per class.
5. Wire v2 classes into `solution.py` once each beats its v1 on the dev labels; T4 timing run; website.

**C3897 relabel check list (27 Sep)**: jaywalking v2 moments not in the labels, for the lead to rewatch.
Walking on the road (likely missed labels): 1:15.3–1:35.7 between the triangle islands; 2:21.6–2:45.7 top-right
corner along the kerb (2:27–2:36 partly the grass strip = zone error); 2:46.6–2:49.9 far road; 4:12.6–4:13.9;
4:33.9–4:43.5 far road. Standing at kerbs (likely our false positives, not labels): 0:29.8, 0:58.9–1:03.3,
1:56–2:16 left kerb by the zebra, 2:37–3:41 top-right corner (one spot, tracks 893/1656), 2:52 bottom edge.

**Detection check: is SAHI-style tiling worth it for jaywalking?** (`experiments/jaywalking/detect_check.py`,
figures `../figures/07_detection_check/`). On 10 frames of the 5 barely-seen labelled events: people found
1280 px 358, 1920 px 471, 8 tiles 621; counted on the carriageway 13 / 9 / 29. CPU time per frame 1.5 s / 2.6 s /
17.6 s (tiles ≈ 12× the current cost). Looking at the frames: the labelled jaywalkers are **already detected at
1280**; they are missed because they walk right beside the zebra stripes (inside our crossing zone + margin), not
because they are small. Tiling mostly adds far pavement people and **drivers seen through windscreens** (new false
"pedestrians" in the road). → No tiling for jaywalking. Label question: people walking just beside the zebra were
labelled jaywalking (C3897 0:31–0:34), which differs from the earlier "a metre off the stripes = crossing" note.

**Decision (27 Sep): beside the zebra = jaywalking** (PDF: "outside a crossing"; the PDF is the main source). So the
crossing zone gets no extra tolerance margin; jaywalking v2 must be retuned with a smaller/zero margin once C3897 is
relabelled. The lead checks whether the exported C3897 file lacks labels that exist in the tool (file vs UI).
**How we are tested** (task PDF): hidden test videos from the same camera and angle, same resolution/fps/viewpoint as
the samples, annotated by the organisers; they run the unchanged `run_submission.py` offline on a T4-class GPU
(`pip install -r requirements.txt`, weights fetched once by `weights/download.sh`) and score with the published
`evaluate.py` — "the same pipeline you can run locally; only the ground truth differs".

**Label export gap found (27 Sep):** the lead sees 14 jaywalking segments in C3897 but the exported file has 8 —
6 are suggestions never accepted (faded), and Export writes only accepted segments. Fix: accept them (A) and export
again; the tool now warns on export when unaccepted suggestions remain, bundles rebuilt. Dev scores so far used the
8-segment file, so C3897 jaywalking "false positives" partly are these unexported labels.
