# How we label the sample videos (dev set)

Without labels every threshold is a guess. The dev set is our own `labels.json` in the exact ground-truth format,
scored with `python evaluate.py --pred predictions_samples.json --gt labels.json --per-video`.

## Workflow (about 1–1.5 h per 5-minute video)
1. Open `tools/label_tool.html` (serve the repo: `python -m http.server 8765`, then http://localhost:8765/tools/label_tool.html).
2. **Open video…** → pick the proxy (`proxies/C3897.mp4`, 1280 wide, same timing as the 4K original).
   Rename is not needed, but the name in the export must match the real file name (`C3897.MP4`); the proxies share the
   stem, so rename the key in `labels.json` or open the file renamed to `.MP4`.
3. **Import…** → `runs/<video>/proposals.json` (rule suggestions, dashed).
4. Pass 1, proposals: press **N** to jump to the next one, watch it, then **A** accept, or fix edges with **[** / **]**
   at the right frame, or **Del** delete.
5. Pass 2, watch the whole video at 2× (**.** / **,** change speed). For anything missed: pick the class key,
   **I** at the start frame, **O** at the end frame. Use **←/→** for single frames at boundaries.
6. **Export labels.json**, commit it under `labels/<video>.json`.

## Conventions (from the task; the tool shows the definition of the selected class)
- One segment = one contiguous event of one class. Two things at once = two segments.
- Same class at the same time (e.g. two jaywalkers) = **one segment covering both** (the tool warns on overlaps).
- Boundaries follow the task table exactly, e.g. failure_to_yield runs from the vehicle *entering* the crossing to it
  *leaving*, not while the pedestrian is there.
- Unsure → still label it, and write it in `labels/notes.md` with the time. We decide as a team.
- Event running past the video end → end = video duration.

## Split
| Video | Length | Light | Who |
|---|---|---|---|
| C3897 | 5:18 | daylight | |
| C3902 | 5:18 | dusk | |
| C3905 | 2:08 | dusk, jam at the end | |

Two people labelling the same 1-minute stretch once and comparing is the quickest way to agree on boundaries.
