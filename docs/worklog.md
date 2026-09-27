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
3. ~~stopped_vehicle v2~~ done (0.83).
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
Added an **"Accept all suggestions"** button (current video, Ctrl+Z undoes) at the lead's request — the lead had
deleted every wrong suggestion, so all remaining ones are labels. Updated tool copied into the lead's unzipped
`label_bundles/label_C3897/label_C3897/` (old page kept as label_tool.old.html; work stays in browser storage).

**C3897 labels v2** (`label_bundles/labels_C3897 v2 merged.json`, from the lead's export
`labels_C3897-accepted ssuggestion.json` after "Accept all"; same-class overlaps merged): 34 events — jaywalking 14
(was 8), failure_to_yield 6, solid_line_crossing 6, congestion 3, stopped_vehicle 2, stop_line 1, near_miss 1,
illegal_turn 1. Official format check VALID. To check: the 3 accepted congestion suggestions (partial-lane rule).
**Jaywalking re-scored on complete labels:** old rule 0.32; **v2 with the settings chosen before the new labels
0.47** (C3897 0.51, C3902 0.42, C3905 0.52); best case after re-tuning 0.50; leave-one-video-out 0.41 (the bigger
search overfits — kept the earlier settings). Margin 0 (beside-the-zebra decision) was not chosen by the search;
0.25 × height still wins, likely because foot points are noisy near the stripes.

**stopped_vehicle v2** (`src/v2/stopped_vehicle.py`, `experiments/stopped_vehicle/`, figures
`../figures/08_stopped_vehicle/`):
- Diagnosis: all 4 labelled stopped vehicles are cars **parked at the far kerb** (by the bus stop), standing minutes
  to the whole video while traffic flows past. v1 missed them: its "≥ 4 other still vehicles = signal queue" filter is
  almost always true somewhere in the frame, and track IDs switch over minutes.
- Design (research: AI City stalled-vehicle winners, ATSPM queue logic): static groups from **near-still vehicle
  boxes linked by overlap (IoU ≥ 0.7) with the group's median box, ignoring track IDs**, surviving 10 s unseen;
  ≥ 10 s, on the carriageway, not parking/bus bay, not waiting inside the junction; **queue = other still vehicles
  within 3 widths for ≥ 50 % of its stop AND driving off within 6 s of it** (a queue stands and leaves together; two
  parked cars leave at unrelated times); stops > 120 s (a signal cycle) are never a queue; starts/ends within 2 s of
  the video edges snap to 0 / the end; simultaneous stops merge into one segment.
- Iterations: naive grouping → queues everywhere; + "leave together" test only → 0.74 (junction waits, far queues);
  + neighbours-stand-with-it test → 0.57 (two parked cars next to each other rejected as a queue);
  + both conditions → **0.83**.
- Result: **old 0.00 → v2 0.83** (tIoU 0.3: 4/4 found, 0 false; 0.5 / 0.7: 3 of 4 — C3897's short 0:00.6–0:25.9
  comes out as 0:07.5–0:41.5). Runs in seconds (vectorised grouping).
- Only 4 labelled events, so no fine-tuning: one-at-a-time sensitivity keeps the score in 0.67–0.86
  (`sensitivity.py`), defaults kept.

**failure_to_yield v2 — in progress (27 Sep, 04:55)** (`src/v2/failure_to_yield.py`, `experiments/failure_to_yield/`,
figures `../figures/09_failure_to_yield/`). 11 labelled events (C3897 6, C3902 3, C3905 2), almost all on
crosswalk_1; 4 of C3897's are 0.6 s long = old-rule timings accepted via "Accept all" (treat their boundaries as weak).
- Old rule: F1 0.10 (6/11 found at 0.3, ~80 false).
- v2 "lane" mode (pedestrian within 1 vehicle width sideways of the vehicle's path): 0.08, 114 predictions.
- v2 "pet" mode (pedestrian spot covered by the vehicle's bottom-band footprint within 2 s): 0.098, 77 predictions.
- Looked at detections: 0:12.3 C3902 is a real one (car passes in front of people on the zebra). The false ones
  (C3902 0:29.5, 1:48.1) are **pedestrians standing on the pavement / kerb just beyond the crossing end**, picked up
  by the 20 px STEP margin around the crossing plus the widened footprint.
- Next to try: STEP 0 (pedestrian strictly on the crossing polygon and on the carriageway), smaller EXPAND, PET 1–3 s,
  pedestrian moving (not standing), grid + leave-one-video-out; then per-crossing checks.

### Overnight plan (lead asleep, autonomous run from 05:20)
Order: finish failure_to_yield → congestion → stop_line fix → solid_line_crossing → near_miss → illegal_turn →
decide wrong_way / red_light / accident / illegal_u_turn / road_obstacle / fire_smoke (emit or not) → integrate v2
into solution.py and score everything together against the 0.094 baseline.

**failure_to_yield v2 (27 Sep, overnight loop, part 1)** (`src/v2/failure_to_yield.py`, `experiments/failure_to_yield/`,
figures `../figures/09_failure_to_yield/`). 11 labelled events, almost all on crosswalk_1; 4 of C3897's are 0.6 s
old-rule timings accepted via "Accept all" (weak boundaries).
- Old rule 0.10 (~80 false). v2 "lane" test (pedestrian within 1 width of the vehicle path) 0.08, 114 predictions;
  v2 PET test (pedestrian spot covered by the vehicle's bottom-band footprint within PET s) 0.098, 77 predictions.
- Looked at detections: false ones were pedestrians standing on the pavement just beyond the crossing end.
- Added pedestrian-must-be-moving filter + cached rider filtering; grid search (`search.py`): **best 0.21 on all 3,
  0.17 leave-one-video-out**, 40 predictions (9 of 11 labelled found in some form). Stable choices across folds:
  EXPAND 0, PET 2 s, PED_MOVING 0.3 — set as defaults.
- Remaining false alarms: 8 on crosswalk_3 (no labelled event there at all), many on crosswalk_1 in C3905.
  Rendered examples (`render.py`): fp_cw3_*, fp_cw1_* — to inspect next.
- Next: inspect those, then consider one-sided PET (pedestrian at the spot *before* the car), per-crossing checks.

**stop_line v2** (`src/v2/stop_line.py`, `experiments/stop_line/evaluate_v2.py`): the vehicle must **arrive past the
line on red** (light red when it first gets past the line; or when it stops, if it was already past at its first
sighting). Fixes the case found while labelling (crossed on green, then held past the line by turning cars).
**v1 0.67 → v2 0.86** on the dev labels (3 labelled events, all found at every tIoU; false alarms 3 → 1, the last one
at the very end of C3902). "not red but not green" (amber) as the arrival condition scored 0.75 → kept "red".
Note: C3897's stop_line label is an accepted old-rule suggestion (its timing is v1's).

**congestion v2** (`src/v2/congestion.py`, `experiments/congestion/`): new per-direction test (vehicles in each drawn
approach: enough of them, most standing, even the faster ones crawling; + junction as an area) — best 0.30 on all 3,
**0.19 leave-one-video-out**, not better than the v1 junction rule (0.29 on the updated labels). v1 + v2 gap bridging:
gap 3–10 s → **0.36**, gap 20 s → 0.52 but that gain is all C3897, whose congestion labels are accepted v1
suggestions (biased towards v1 timings; C3902/C3905 unchanged). Default = v1 junction rule + 10 s bridging (0.36);
per-direction mode kept for better labels. C3902's labelled jams are approach_1 spilling onto the crossing and junction.

**solid_line_crossing v2** (`src/v2/solid_line.py`, `experiments/solid_line/`): 9 labels. Box-corner side tests →
388 raw crossings (a box is wider than the car from this oblique camera), best 0.11, **0.0 leave-one-video-out**.
Footprint-centre version (side change of the centre, event = crossing −1 s … +0.5 s; straddling ≥ 8 s counts) → best
0.14 on all 3 with the solid stretch = first 20 % of each drawn line from the stop line, still **0.0 LOVO**. Weak, but
the class occurs in all 3 samples, so it is emitted (a class present in the test set is scored whether we predict it or
not; a weak detector can only add). Needs the redrawn solid lines (lead's pending fix) — redo then.

**Emission rule (decided 27 Sep):** emit every class that appears in our sample labels (it is almost surely in the test
set; any non-zero F1 adds); do not emit classes never seen in the samples unless a detector is reliable (an absent
class that we predict adds a zero to the macro average).

**near_miss v2 — not emitted** (`src/v2/near_miss.py`, `experiments/near_miss/search.py`): built on the Part B
estimator (strongest pair's DRAC per update; conflict runs; optional hard-braking check). The two labelled near misses
(slow turning conflicts in the junction) barely register — max DRAC 1.8 (C3897) and 9.1 (C3905) around them, while
normal traffic peaks at 21–23 → best F1 0.02. Decision: do not emit near_miss (≈0 F1 adds nothing if present, and an
absent class we predict adds a zero). Needs a different signal (e.g. sudden deceleration / heading change of a turning
vehicle next to another) — later.

**illegal_turn v2** (`src/v2/illegal_turn.py`, `experiments/illegal_turn/`): origin–destination + lane of origin.
Every left turn from approach_1 to the bottom-left road (approach_3) in the 3 videos comes from lane 0 (leftmost,
lanes counted with the drawn dividers) — except exactly the two labelled illegal turns (C3897 #1147, C3902 #2620),
which turned left from lane 1. Rule: left turn to the bottom-left road from lane ≥ 1 = turn from the wrong lane.
Event = leave-approach − 1.5 s … reach exit road + 0.5 s (timing sweep running).
illegal_turn timing sweep (PRE 0.5–2.5 s, POST 0–3 s, exit tolerance 200–800 px): F1 1.0 everywhere (2/2, no false
alarms) — insensitive; defaults kept (PRE 1.5, POST 0.5, 400 px).

**All v2 classes together (`src/v2/pipeline.py`, `experiments/combined/evaluate_all.py`), dev labels, official metric:**
**Score A v1 0.138 (on the updated labels) → v2 0.483** over 8 classes: illegal_turn 1.00, stop_line 0.86,
stopped_vehicle 0.83, jaywalking 0.47, congestion 0.36, failure_to_yield 0.21, solid_line_crossing 0.14, near_miss 0
(labelled, not emitted). Optimistic (tuned on the same labels; per-class leave-one-video-out numbers are lower), but
the gain is large and in every class. v1's red_light / wrong_way false positives are gone (not emitted).
**Speed** (needed: tracking 1.3× + Part B ≈ 1× + event layer must fit 3×): first v2 run took 150–224 s per 5-min video
on this CPU (up to 0.7×). Fixes: shared vectorised rider test (`src/v2/common.py`), illegal_turn exit zone by distance
transform instead of a 4K dilation with a 801 px kernel (80 s → 1 s), stopped_vehicle frame loop on arrays instead of
pandas indexing (58 s → 18 s). Scores unchanged.
`solution.py` now uses the v2 event layer by default (`EVENT_LAYER=v1` switches back); end-to-end harness check running.

**failure_to_yield v2, round 2:** added walking-speed cap (riders whose vehicle box was missed), partial-box filter
(pedestrian box < 0.5 × vehicle box height: heads of people behind the car), crossing skip option; grid with cached
pedestrians. **0.21 → 0.27 on all 3 (0.26 with the chosen defaults), leave-one-video-out 0.17 → 0.22**, 21–30
predictions for 11 labels. Every fold chose to skip crosswalk_3 (never labelled there in 12.6 min; 8 false alarms
there) — adopted, **flagged for the lead to review** (label-driven, not from the task definition).
Defaults: STEP 10 px, PET 2 s, pedestrian speed 0.2–1.5 heights/s, partial < 0.5, vehicle moving ≥ 0.4.

**End-to-end check:** `run_submission.py` on the 21 s 4K test clip with the v2 layer inside `solution.py`: runs
without errors, 6 events (v2); the harness then drops them because this CPU laptop needs 583 s for Part A (budget
63 s) — same as v1 on CPU; the real timing test needs the T4.

**Final combined v2 (all classes, current defaults): Score A 0.490** on the dev labels (v1 0.138); event layer runs in
5–13 s per 5-min video on this CPU (~0.04 × duration).
**README rewritten** to the task's required contents: install/run incl. weights, approach (architecture, models,
learned vs rule-based, per-class method + dev score), datasets/licences, seeds and non-determinism, team and roles
(teammates' roles marked "to confirm").

### Next actions (updated 27 Sep, morning)
1. **Lead:** review the crosswalk_3 skip in failure_to_yield (label-driven); confirm teammates' roles in README.
2. **Lead:** redraw the solid lines (only the stretch next to the stop line is solid) and the grass strip in the
   road zone → then re-run `experiments/solid_line/search_centre.py` and jaywalking.
3. **On the lab GPU / T4:** `python run_submission.py --videos samples --out predictions_samples.json` → commit it
   (required file; reproducibility is 25 % of the code score); also gives the real timing (must be < 3 × duration).
4. Label C3896 (4th sample, suggestions ready) to check v2 on unseen video.
5. Website (25 % of the elimination score) — not started.
6. Make the repo public and tag the submission commit before the deadline (lead's decision).

**Sanity run on C3896 (4th sample, unlabelled, not used for tuning):** 28 events — stopped_vehicle 1 (0:00–5:05,
parked car), jaywalking 4, failure_to_yield 8 (0.5–4 s), congestion 4, stop_line 1, illegal_turn 3 (two of them
15–22 s long: turners waiting in the junction?), solid_line_crossing 7 (several 40–68 s stretches from queued cars
standing on the lane dividers — the known weak class). Nothing broken; v2 suggestions saved as
`../runs/jaywalking_v2/proposals_v2_C3896.json` (not put into the labelling bundle, to keep future C3896 labels
unbiased — lead's choice).

## Website and live demo (27 Sep, day)
Task PDF deliverables checked: repo (done except predictions_samples.json, T4 timing, public + tag), website with 7
required sections and a rubric (live demo 30 %, sample visualisations 20 %, EDA 15 %, approach+report 15 %, team 10 %,
design/extras 10 %), one-page report. AI Lab proposal already allows publishing annotated visualisations (not raw video).
- `tools/render_results.py`: annotated playback of all 4 samples (boxes by class, active events, timeline strip,
  risk) + per-video events and Part B risk curve (causal estimator replayed on the saved tracks).
- `tools/make_site_data.py`: EDA — video facts, counts per class over time, signal_5 timeline, motion heat maps,
  trajectories coloured by direction with zones; our dev labels for the timelines.
- `tools/make_site_media.py`: example frame per class, failure-case images.
- `demo/site/`: the website (single page, no frameworks: approach + pipeline diagram, EDA, results with click-to-seek
  timelines and prediction-vs-label rows, error analysis, live demo, report, team, links; works on a phone).
- Live demo: first a Gradio app, dropped at the lead's request (own UI instead). `demo/server.py` (standard library:
  static site with Range support + `POST /api/jobs` upload, `GET /api/jobs/<id>` progress/results, one job at a time,
  queue position, 1 h cleanup) + `demo/process.py` (same stages as solution.py; 4K-canonical coordinates; YOLO26s
  ≤ ~50 s clips else YOLO26n, 960 px, 5 fps; annotated playback, events, risk). Tested end to end locally: 21 s 4K
  clip → 4 events, playable video, timeline, risk, table in 47 s on this laptop.
- Hugging Face: Docker/Gradio Spaces on free CPU now need PRO; only static Spaces are free → hosting decision
  deferred by the lead ("make it super good locally first"). `tools/build_space.py` builds the Docker Space anyway.
- Demo polish: aligned-zones preview of the upload, stage checklist + time-left estimate, drag and drop, class chips;
  tested end to end again (21 s clip → 4 events, zones fit).
- Site additions: signal cycle (red 33–42 s, green 27–41 s, cycle ≈ 75–80 s; flicker < 3 s ignored), speed
  distributions, ablations (detector/tracker benchmark + event-layer variants), operator dashboard (116 events in
  18 min, 6.3/min), 12 s hero loop; C3896 is daylight (brightness 96 like C3897), not evening; no horizontal overflow
  at 375 px.
- Demo input: any size/resolution now (the PDF lets us state our limits; lead's choice). FFmpeg decodes 5 fps scaled
  to 1920 wide (multi-threaded, any input); only a 10 GB disk-safety cap. Tested on the 21 s 4K clip.
- Diagram: `docs/pipeline.excalidraw` (+ `boards/solution-pipeline.excalidraw`), generated by
  `tools/make_pipeline_excalidraw.py`; rendered with Excalidraw's own exporter to `docs/pipeline.svg`, used on the site.
- No GPU until the deadline (AI Lab access not before 23:59) → `tools/make_predictions_samples.py` writes
  predictions_samples.json on CPU with solution.py's exact steps: Part A = event layer on the saved YOLO26m@1280
  every-3rd-frame tracks (same settings as solution.py), lights read every 3rd frame, first-frame alignment;
  Part B = solution.RiskEstimator fed every frame like run_submission (budget lifted for CPU). Running (~1–2 h).
- **predictions_samples.json done** (CPU, `tools/make_predictions_samples.py`): 4 videos, 120 events, risk for every
  frame (max 0.28–0.43), `evaluate.py --validate-only` VALID, committed; README explains how it was produced.
- Demo test clips (`../demo_test_clips/`, from the samples, team only): 30 s day 1280, 60 s dusk 480p, 45 s jam 1080p,
  2 min day 1280. The 480p clip found a bug: odd output height (629 px) broke the H.264 writer → heights rounded to
  even; zones still aligned at 480p; 60 s clip in 37.5 s.
- **Team update (27 Sep):** Muhammadjon Xalimov replaces Dilshodbek Tolibjonov; team = Davlatyor Abduloxunov (lead),
  Muhammadjon Xalimov, Behruz Xasanov. Website Team section with photos (Behruz: initials placeholder until his photo
  arrives), GitHub / LinkedIn / portfolio links; README team table updated (student IDs removed from the public README).
  Still needed: each member's actual contributions and previous projects.
- Team section redesigned (reference layout: large portrait, spaced uppercase role line, bold name, short text), full uncropped 3:4 photos; Bexruz (with x) Xasanov's photo added and spelling fixed.
