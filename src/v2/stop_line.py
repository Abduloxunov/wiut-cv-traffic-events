"""stop_line v2: a vehicle stops past the stop line on red without entering the intersection
(start: vehicle stops, end: signal turns green).

Fix over v1 (found while labelling): v1 flagged any vehicle standing past the line while the light was red, so a
car that crossed on green and was then held past the line by a queue of turning cars was flagged once the light
turned red. v2 requires the vehicle to *arrive* past the line on red: the light is red (not green) when the vehicle
first gets past the line, or when it comes to a stop there if it was already past at the start of the video.
Only stop lines whose signal head we can read are used (stop_line_1 <- signal_5).
"""
import numpy as np

from rules import _side, _state_at, _stop_lines, STILL, VEHICLES
from scene import lookup
from v2.segments import runs, union

DEFAULTS = dict(
    PAST=0.4,        # box heights past the line (front over the line)
    FAR=2.5,         # box heights: farther past = already in the junction approach, not a stop-line case
    MIN_STOP=1.0,    # s standing still past the line
    ARRIVE="red",    # light state required when the vehicle gets past the line ("red" or "not_green")
    GAP=1.0,
)


class StopLine:
    def __init__(self, scene, signals, **params):
        self.p = {**DEFAULTS, **params}
        self.scene, self.signals = scene, signals

    def detect(self, df):
        p = self.p
        veh = df[df.cls.isin(VEHICLES)]
        segs = []
        for name, a, b, times, states in _stop_lines(self.scene, self.signals):
            d = b - a
            pts = veh[["gx", "gy"]].to_numpy()
            side = _side(pts, a, b)
            approach = np.sign(np.median(side[(veh.speed < STILL).to_numpy()])) or 1.0
            proj = ((pts - a) @ d) / (d @ d)
            dist = np.abs(d[0] * (pts[:, 1] - a[1]) - d[1] * (pts[:, 0] - a[0])) / np.linalg.norm(d)
            bh = veh.bh.to_numpy()
            past = (side == -approach) & (proj > 0) & (proj < 1) & (dist > p["PAST"] * bh) & (dist < p["FAR"] * bh) \
                & ~lookup(self.scene.junction, veh.gx, veh.gy)
            v = veh.assign(past=past, still=(veh.speed.fillna(0) < STILL).to_numpy())
            for tid, t in v.groupby("track_id"):
                if not t.past.any():
                    continue
                t = t.sort_values("t_sec")
                arrive = t.t_sec[t.past].iloc[0]
                for s, e in runs((t.past & t.still).to_numpy(), t.t_sec.to_numpy(), max_gap=p["GAP"]):
                    if e - s < p["MIN_STOP"]:
                        continue
                    when = arrive if arrive > t.t_sec.iloc[0] else s      # already past at its first sighting
                    st = _state_at(times, states, when)
                    ok = st == "red" if p["ARRIVE"] == "red" else st != "green"
                    if not ok:
                        continue
                    green = times[(times > s) & (states == "green")]
                    segs.append((s, float(green[0]) if len(green) else float(t.t_sec.iloc[-1])))
        return [(a, b, "") for a, b in union(segs)]
