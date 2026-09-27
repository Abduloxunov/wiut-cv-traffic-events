"""near_miss v2: sharp braking or swerving to avoid a collision, no contact
(start: onset of the evasive action, end: road users are clear of each other).

Design (research: surrogate safety measures -- TTC / DRAC conflicts from trajectories; the Part B estimator
src/risk.py already finds pairs that are apart, closing fast and on a collision course):
  1. replay the causal risk estimator over the tracks; per update keep the strongest pair's DRAC (deceleration rate
     needed to avoid the collision, in object sizes / s^2) and that pair;
  2. conflict stretches = runs where DRAC >= DRAC_MIN for at least MIN_DUR, bridged over GAP;
  3. evasive action (optional, EVASIVE): at least one of the pair decelerates by >= DECEL sizes/s within the stretch
     (hard braking) -- a conflict nobody reacts to is not a *near miss* by the definition;
  4. event = [start - PRE, end + POST] (onset of the evasive action ... until they are clear).
"""
import numpy as np

from risk import CollisionRisk
from v2.segments import union

DEFAULTS = dict(
    DRAC_MIN=10.0,   # sizes / s^2 (DRAC_HALF = 20 maps to Part B score 0.5)
    MIN_DUR=0.3,     # s
    GAP=1.0,         # s
    EVASIVE=False,
    DECEL=0.8,       # sizes / s: speed drop of one of the pair inside the stretch
    PRE=0.5,
    POST=1.0,
)


def replay(df, scene):
    """[(t, drac, (id_a, id_b))] per processed frame: the strongest pair of the causal risk estimator (0 if none)."""
    est = CollisionRisk(scene)
    cols = ["frame", "t_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"]
    out = []
    for f, g in df[cols].groupby("frame", sort=True):
        t = float(g.t_sec.iloc[0])
        _, best = est.update(t, g.itertuples(index=False, name=None))
        out.append((t, best["drac"] if best else 0.0, best["ids"] if best else None))
    return out


class NearMiss:
    def __init__(self, scene, **params):
        self.p = {**DEFAULTS, **params}
        self.scene = scene

    def _speed(self, df, tid, t0, t1):
        t = df[(df.track_id == tid) & (df.t_sec >= t0 - 0.5) & (df.t_sec <= t1 + 0.5)].sort_values("t_sec")
        if len(t) < 4:
            return None
        size = np.sqrt((t.x2 - t.x1) * (t.y2 - t.y1)).mean()
        return np.hypot(t.vx.fillna(0), t.vy.fillna(0)).to_numpy() / max(size, 1.0)

    def detect(self, df, trace=None):
        p = self.p
        trace = replay(df, self.scene) if trace is None else trace
        times = np.array([x[0] for x in trace])
        drac = np.array([x[1] for x in trace])
        hot = drac >= p["DRAC_MIN"]
        segs = []
        start = None
        for k in range(len(times) + 1):
            on = k < len(times) and hot[k]
            if on and start is None:
                start = k
            if not on and start is not None:
                a, b = times[start], times[k - 1]
                pairs = {trace[i][2] for i in range(start, k) if trace[i][2]}
                ok = b - a >= p["MIN_DUR"]
                if ok and p["EVASIVE"]:
                    ok = False
                    for pair in pairs:
                        for tid in pair:
                            sp = self._speed(df, tid, a, b + 1.0)
                            if sp is not None and sp.max() - sp[np.argmax(sp):].min() >= p["DECEL"]:
                                ok = True
                if ok:
                    segs.append((max(0.0, a - p["PRE"]), b + p["POST"]))
                start = None
        return [(a, b, "") for a, b in union(segs, gap=p["GAP"])]
