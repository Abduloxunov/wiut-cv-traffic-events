"""red_light v2: a vehicle crosses the stop line while its signal is red
(start: the front of the vehicle crosses the stop line, end: the vehicle leaves the intersection or the frame).

Design (research: stop-line crossing + debounced red light is the universal rule in red-light systems; the known
pitfalls are vehicles entering on amber and early starters just before green):
  * crossing = the vehicle's ground point (bottom centre of the box = its front, since traffic on this approach drives
    towards the camera) changes side of the stop line, from the approach side, inside the line's extent;
  * the light (signal_5 for stop_line_1) must have been red for at least RED_SETTLE s: at dusk the amber lamp glows
    orange-red and is read as red (C3902 2:35.9), and amber lasts ~3 s, so a settled red is past any amber;
  * and must stay red for at least RED_AHEAD s after the crossing: drivers moving off a moment before green are not
    red-light runners;
  * the vehicle must actually drive on (not stop past the line: that is stop_line);
  * end = it leaves the junction, or first comes to rest after crossing it (held in traffic at the exit).
Checked on all four sample videos: fires once, on the one real runner (C3896 1:18.9, red for 12.8 s, pedestrians
crossing on their green), and on none of the 26 other non-green crossings (amber, flashing green, early starts).
"""
import numpy as np

from rules import VEHICLES, _red_ahead, _red_since, _side, _stop_lines
from scene import lookup
from v2.segments import union

DEFAULTS = dict(
    RED_SETTLE=4.0,   # s the light has been red before the crossing
    RED_AHEAD=2.0,    # s it stays red after the crossing
    MIN_SPEED=0.5,    # box heights / s at the crossing: driving through
    MIN_AFTER=1.0,    # s the vehicle is followed after the crossing
    STOPPED=0.2,      # box heights / s: first rest after the crossing ends the event
)


class RedLight:
    def __init__(self, scene, signals, **params):
        self.p = {**DEFAULTS, **params}
        self.scene, self.signals = scene, signals

    def detect(self, df, explain=False):
        p = self.p
        veh = df[df.cls.isin(VEHICLES)]
        after = self.scene.junction | self.scene.mask({"crosswalk"})
        segs, why = [], []
        for name, a, b, times, states in _stop_lines(self.scene, self.signals):
            d = b - a
            crossings = []
            for tid, t in veh.groupby("track_id"):
                t = t.sort_values("t_sec")
                pts = t[["gx", "gy"]].to_numpy()
                side = _side(pts, a, b)
                proj = ((pts - a) @ d) / (d @ d)
                for i in range(1, len(pts)):
                    if side[i - 1] != side[i] and side[i] != 0 and -0.05 <= proj[i] <= 1.05:
                        crossings.append((tid, i, side[i - 1], t))
            if not crossings:
                continue
            approach = np.sign(sum(c[2] for c in crossings))
            for tid, i, from_side, t in crossings:
                tc = float(t.t_sec.iloc[i])
                if from_side != approach:
                    continue
                if _red_since(times, states, tc) < p["RED_SETTLE"] or _red_ahead(times, states, tc) < p["RED_AHEAD"]:
                    continue
                if not (t.speed.iloc[max(0, i - 3):i + 3].fillna(0).max() >= p["MIN_SPEED"]):
                    continue
                rest = t.iloc[i:]
                if rest.t_sec.iloc[-1] - tc < p["MIN_AFTER"]:
                    continue
                inside = lookup(after, rest.gx, rest.gy)
                end = float(rest.t_sec[inside].max()) if inside.any() else float(rest.t_sec.iloc[-1])
                # through the junction and then held in traffic at its exit: it has left the conflict area when it
                # first comes to rest (C3896: crossed at 1:18.9, stopped at the exit at ~1:24, left the frame 1:51)
                still = rest[(rest.t_sec > tc + 1.0) & (rest.speed.fillna(1) < p["STOPPED"])]
                if len(still):
                    end = min(end, float(still.t_sec.iloc[0]))
                segs.append((tc, max(end, tc + 0.5)))
                why.append((tc, end, name, tid, round(_red_since(times, states, tc), 1)))
        res = [(s, e, "") for s, e in union(segs)]
        return (res, why) if explain else res
