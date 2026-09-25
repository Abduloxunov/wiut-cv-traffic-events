# The 14 classes: what tracking alone can do, what needs a learned model

Research basis: public violation-detection repos (detector + tracker + zone rules), AI City Challenge traffic-anomaly
winners (background modelling), the ACCIDENT @ CVPR 2026 zero-shot CCTV accident challenge (teams' papers and repos),
the Nexar dashcam collision challenge (Kaggle 2025), surrogate-safety literature (TTC, PET, DRAC). Links in
`docs/research.md`.

## A. Solvable with tracks + zones + signal state (no training)
| Class | What the rule needs | Status |
|---|---|---|
| `jaywalking` | person's feet on road outside crossings/pavements/islands | v1 in `src/rules.py` |
| `failure_to_yield` | moving vehicle inside a crossing, pedestrian on it near the vehicle's path | v1 |
| `stopped_vehicle` | vehicle still ≥ 10 s on the road, outside bus bay / parking, not in a signal queue | v1 (+ background modelling planned) |
| `congestion` | many vehicles standing still across a direction / inside the junction | v1 |
| `wrong_way` | vehicle moving against the lane arrows outside the junction | v1 |
| `red_light` | stop-line crossing while its signal has been red ≥ 0.5 s and stays red ≥ 1 s | v1, needs the signal reader |
| `stop_line` | vehicle standing clearly past the stop line on red | v1, needs the signal reader |
| `solid_line_crossing` | track crosses a drawn solid line laterally | not yet (lines are drawn) |
| `illegal_turn`, `illegal_u_turn` | entry → exit approach per track vs a list of banned movements | not yet; needs the banned list from the team |

## B. Need a learned or open-vocabulary model on top of tracking
| Class | Best evidence found | Plan |
|---|---|---|
| `accident` | ACCIDENT@CVPR 2026 (real CCTV, zero-shot): physics heuristics on tracks (overlap, trajectory intersection, approach speed) beat learned relational models; frame-difference z-score peaks give the impact time; the top pipelines add a VLM on candidate windows | track heuristic + frame-difference peak for the start time; verifier optional |
| `near_miss` | surrogate safety measures (TTC < 1.5 s serious, DRAC) + evasive action (hard braking, swerve) | from the Part B risk signal + deceleration spikes; emit only if precise |
| `road_obstacle` | open-vocabulary detectors (YOLOE / YOLO-World) find debris, animals, boxes by text prompt without training; background modelling finds new static objects | only with strong evidence, else never emitted |
| `fire_smoke` | D-Fire-trained YOLO or open-vocabulary "smoke"/"fire"; dusk headlights/brake lights are the main false positives | only with strong evidence, else never emitted |

Why "never emitted" is a valid choice: a class predicted but absent from the test set scores 0 and joins the macro
average; a class present but not predicted also scores 0. For rare classes with a weak detector, silence is safer.
