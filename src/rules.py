"""Rule-based traffic events from tracks + scene zones.

Every rule takes the track table (see `add_motion`) and a `Scene` in the same video pixels and returns
raw segments [(start_sec, end_sec, why)]. `detect` runs the enabled rules and merges same-class segments
(the ground truth reports simultaneous same-class events as one segment).
"""
import numpy as np
import pandas as pd

from scene import lookup

VEHICLES = {"car", "bus", "truck", "motorcycle"}
STILL = 0.12        # box heights per second below which a vehicle counts as standing still
MOVING = 0.5        # box heights per second above which a vehicle counts as clearly moving
EDGE = 0.01         # ignore people within this fraction of the frame border (half-visible, unreliable feet)


def add_motion(df, fps):
    """Majority class per track, ground point, and speed / velocity over a +-1 s window."""
    df = df.copy()
    df["cls"] = df.groupby("track_id").cls.transform(lambda s: s.mode().iloc[0])
    df["gx"], df["gy"] = (df.x1 + df.x2) / 2, df.y2
    df["bh"] = (df.y2 - df.y1).clip(lower=8)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    step = df.groupby("track_id").frame.diff().median()
    k = max(1, int(round(fps / (step if step and step > 0 else 3))))
    g = df.groupby("track_id")
    dx, dy = g.gx.shift(-k) - g.gx.shift(k), g.gy.shift(-k) - g.gy.shift(k)
    dt = (g.frame.shift(-k) - g.frame.shift(k)) / fps
    df["vx"], df["vy"] = dx / dt, dy / dt
    df["speed"] = np.hypot(df.vx, df.vy) / df.bh
    return df


def runs(flags, times, min_len, max_gap=0.6):
    """Contiguous True stretches of a time series -> [(start, end)]; gaps up to max_gap are bridged."""
    out, start, last = [], None, None
    for t, f in zip(times, flags):
        if not f:
            continue
        if start is None or t - last > max_gap:
            if start is not None and last - start >= min_len:
                out.append((start, last))
            start = t
        last = t
    if start is not None and last - start >= min_len:
        out.append((start, last))
    return out


def merge(segments, gap=1.0):
    """Union overlapping / near same-class segments."""
    out = []
    for s, e, why in sorted(segments):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2].add(why)
        else:
            out.append([s, e, {why}])
    return [(s, e, ", ".join(sorted(w))[:80]) for s, e, w in out]


# ------------------------------------------------------------------ rules
def stopped_vehicle(df, scene):
    veh = df[df.cls.isin(VEHICLES)]
    on_road = lookup(scene.road, veh.gx, veh.gy) & ~lookup(scene.exempt_stop, veh.gx, veh.gy)
    still = on_road & (veh.speed < STILL).values
    still_per_frame = veh[still].groupby("frame").size()
    segs = []
    for tid, t in veh[still].groupby("track_id"):
        for s, e in runs(np.ones(len(t), bool), t.t_sec.values, min_len=10.0, max_gap=1.0):
            during = t[(t.t_sec >= s) & (t.t_sec <= e)]
            if still_per_frame.reindex(during.frame).fillna(0).median() >= 4:
                continue  # many others still at the same time: a signal queue, not a stopped vehicle
            segs.append((s, e, f"{t.cls.iloc[0]} #{tid}"))
    return segs


def _inside_vehicle(people, vehicles):
    """True where a person's feet fall inside a vehicle box of the same frame (passengers, riders)."""
    boxes = {f: b[["x1", "y1", "x2", "y2"]].to_numpy() for f, b in vehicles.groupby("frame")}
    out = np.zeros(len(people), bool)
    for i, (f, x, y) in enumerate(zip(people.frame.values, people.gx.values, people.gy.values)):
        b = boxes.get(f)
        if b is not None:
            out[i] = bool(((b[:, 0] <= x) & (x <= b[:, 2]) & (b[:, 1] <= y) & (y <= b[:, 3] + 10)).any())
    return out


def jaywalking(df, scene):
    ppl = df[df.cls == "person"]
    h, w = scene.size
    near_edge = (ppl.gx < EDGE * w) | (ppl.gx > (1 - EDGE) * w) | (ppl.gy > (1 - EDGE) * h)
    on_road = lookup(scene.road, ppl.gx, ppl.gy) & ~lookup(scene.walk_ok, ppl.gx, ppl.gy) & ~near_edge.values
    flag = on_road & ~_inside_vehicle(ppl, df[df.cls.isin(VEHICLES | {"bicycle"})])
    segs = []
    for tid, t in ppl.assign(flag=flag).groupby("track_id"):
        for s, e in runs(t.flag.values, t.t_sec.values, min_len=1.0):
            segs.append((s, e, f"person #{tid}"))
    return segs


def failure_to_yield(df, scene):
    """A moving vehicle inside a crossing while a pedestrian is on that crossing close to its path."""
    ppl = df[df.cls == "person"]
    ppl = ppl[~_inside_vehicle(ppl, df[df.cls.isin(VEHICLES | {"bicycle"})])]
    veh = df[df.cls.isin(VEHICLES)]
    segs = []
    for cw in scene.of_type("crosswalk"):
        m = scene.mask(None, shapes=[cw])
        m_ped = scene.mask(None, margin=15, shapes=[cw])  # "on it or stepping onto it"
        peds = ppl[lookup(m_ped, ppl.gx, ppl.gy)]
        ped_by_frame = {f: g[["gx", "gy"]].to_numpy() for f, g in peds.groupby("frame")}
        v_in = veh[lookup(m, veh.gx, veh.gy)]
        for tid, t in v_in.groupby("track_id"):
            if t.speed.median() < STILL:
                continue  # waiting at the crossing is yielding
            close = []
            for f, x, y, bh in zip(t.frame.values, t.gx.values, t.gy.values, t.bh.values):
                p = ped_by_frame.get(f)
                close.append(p is not None and bool((np.hypot(p[:, 0] - x, p[:, 1] - y) < 4 * bh).any()))
            if not any(close):
                continue
            for s, e in runs(np.ones(len(t), bool), t.t_sec.values, min_len=0.3, max_gap=0.6):
                segs.append((s, e, f"{cw['name']} {t.cls.iloc[0]} #{tid}"))
    return segs


def congestion(df, scene):
    veh = df[df.cls.isin(VEHICLES)]
    stuck = veh[lookup(scene.junction, veh.gx, veh.gy) & (veh.speed < STILL).values]
    per_t = stuck.groupby("t_sec").track_id.nunique()
    times = np.array(sorted(df.t_sec.unique()))
    flag = per_t.reindex(times).fillna(0).values >= 4
    return [(s, e, "vehicles stuck in junction") for s, e in runs(flag, times, min_len=5.0, max_gap=1.0)]


def wrong_way(df, scene):
    arrows = [s["points"] for s in scene.of_type("lane_dir")]
    veh = df[df.cls.isin(VEHICLES) & (df.speed > MOVING)]
    veh = veh[~lookup(scene.junction, veh.gx, veh.gy)]
    if not arrows or veh.empty:
        return []
    P = veh[["gx", "gy"]].to_numpy()
    best_d, best_dir = np.full(len(P), np.inf), np.zeros((len(P), 2))
    for a in arrows:
        seg = a[1] - a[0]
        t = np.clip(((P - a[0]) @ seg) / (seg @ seg), 0, 1)
        d = np.hypot(*(P - (a[0] + t[:, None] * seg)).T)
        closer = d < best_d
        best_d[closer], best_dir[closer] = d[closer], seg / np.linalg.norm(seg)
    v = veh[["vx", "vy"]].to_numpy()
    cos = (v * best_dir).sum(1) / (np.linalg.norm(v, axis=1) + 1e-9)
    segs = []
    for tid, t in veh.assign(against=(cos < -0.7) & (best_d < 250)).groupby("track_id"):
        for s, e in runs(t.against.values, t.t_sec.values, min_len=1.0):
            segs.append((s, e, f"{t.cls.iloc[0]} #{tid}"))
    return segs


# Which readable signal head governs each stop line. Verified from traffic: across the three samples 343 of 351
# vehicles crossed stop_line_1 while signal_5 was green (the rest are red-light candidates).
SIGNAL_FOR_STOP_LINE = {"stop_line_1": "signal_5"}
RED_SETTLE = 0.5    # s the light must already be red, so a vehicle clearing on the change is not flagged
RED_AHEAD = 1.0     # s the light must stay red after the crossing: drivers moving off just before green are not
                    # red-light runners (review of the samples: 3 of 6 raw candidates were such early starts)
PAST_LINE = 0.4     # a stop-line violator's front is at least this many box heights past the line


def signal_timeline(states, name):
    """(times, states) for one signal with 'off' samples filled from the last known state."""
    s = states[["t_sec", name]].copy()
    s[name] = s[name].where(s[name] != "off").ffill().fillna("off")
    return s.t_sec.to_numpy(), s[name].to_numpy()


def _state_at(times, states, t):
    return states[max(0, np.searchsorted(times, t, side="right") - 1)]


def _red_since(times, states, t):
    """Seconds the light has been continuously red at time t (0 if it is not red)."""
    i = max(0, np.searchsorted(times, t, side="right") - 1)
    if states[i] != "red":
        return 0.0
    j = i
    while j > 0 and states[j - 1] == "red":
        j -= 1
    return t - times[j]


def _red_ahead(times, states, t):
    """Seconds the light stays red after time t (0 if it is not red at t)."""
    i = max(0, np.searchsorted(times, t, side="right") - 1)
    if states[i] != "red":
        return 0.0
    j = i
    while j + 1 < len(states) and states[j + 1] == "red":
        j += 1
    return times[j] - t


def _side(points, a, b):
    d = b - a
    return np.sign(d[0] * (points[:, 1] - a[1]) - d[1] * (points[:, 0] - a[0]))


def _stop_lines(scene, signals):
    """(name, a, b, signal times, signal states) for every stop line whose signal we can read."""
    out = []
    if signals is None or signals.empty:
        return out
    for s in scene.of_type("stop_line"):
        sig = SIGNAL_FOR_STOP_LINE.get(s["name"])
        if sig in signals:
            times, states = signal_timeline(signals, sig)
            out.append((s["name"], s["points"][0], s["points"][-1], times, states))
    return out


def red_light(df, scene, signals=None):
    """Vehicle crosses the stop line from its approach side while its signal has been red for RED_SETTLE s.
    Segment: crossing -> vehicle leaves the junction / crossings (or its track ends)."""
    veh = df[df.cls.isin(VEHICLES)]
    after = scene.junction | scene.mask({"crosswalk"})
    segs = []
    for name, a, b, times, states in _stop_lines(scene, signals):
        d = b - a
        crossings = []
        for tid, t in veh.groupby("track_id"):
            p, ts = t[["gx", "gy"]].to_numpy(), t.t_sec.to_numpy()
            side = _side(p, a, b)
            proj = ((p - a) @ d) / (d @ d)
            for i in range(1, len(p)):
                if side[i - 1] != side[i] and side[i] != 0 and -0.05 <= proj[i] <= 1.05:
                    crossings.append((tid, i, side[i - 1], t))
        if not crossings:
            continue
        approach = np.sign(sum(c[2] for c in crossings))  # most vehicles come from the approach side
        for tid, i, from_side, t in crossings:
            tc = t.t_sec.iloc[i]
            if from_side != approach or _red_since(times, states, tc) < RED_SETTLE \
                    or _red_ahead(times, states, tc) < RED_AHEAD:
                continue
            rest = t.iloc[i:]
            inside = lookup(after, rest.gx, rest.gy)
            end = rest.t_sec[inside].max() if inside.any() else rest.t_sec.iloc[-1]
            segs.append((tc, max(end, tc + 0.5), f"{name} {t.cls.iloc[0]} #{tid}"))
    return segs


def stop_line(df, scene, signals=None):
    """Vehicle stands still past the stop line (not yet in the junction) while red; ends when the light turns green."""
    veh = df[df.cls.isin(VEHICLES)]
    segs = []
    for name, a, b, times, states in _stop_lines(scene, signals):
        d = b - a
        p = veh[["gx", "gy"]].to_numpy()
        side = _side(p, a, b)
        # approach side = where vehicles wait at red most of the time
        approach = np.sign(np.median(side[(veh.speed < STILL).to_numpy()])) or 1.0
        proj = ((p - a) @ d) / (d @ d)
        dist = np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])) / np.linalg.norm(d)
        bh = veh.bh.to_numpy()
        past = (side == -approach) & (proj > 0) & (proj < 1) & (dist > PAST_LINE * bh) & (dist < 2.5 * bh) \
            & ~lookup(scene.junction, veh.gx, veh.gy)
        red = np.array([_state_at(times, states, t) == "red" for t in veh.t_sec.to_numpy()])
        flag = past & (veh.speed < STILL).to_numpy() & red
        for tid, t in veh.assign(flag=flag).groupby("track_id"):
            for s, _ in runs(t.flag.values, t.t_sec.values, min_len=1.0, max_gap=1.0):
                green = times[(times > s) & (states == "green")]
                segs.append((s, green[0] if len(green) else t.t_sec.iloc[-1], f"{name} {t.cls.iloc[0]} #{tid}"))
    return segs


RULES = {
    "stopped_vehicle": stopped_vehicle,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "congestion": congestion,
    "wrong_way": wrong_way,
    "red_light": red_light,
    "stop_line": stop_line,
}
NEEDS_SIGNALS = {"red_light", "stop_line"}


def detect(df, scene, duration, classes=tuple(RULES), signals=None):
    """Run the enabled rules -> [(start, end, label, why)], merged per class, clipped to the video.
    `signals`: DataFrame with t_sec and one state column per readable signal head (see signals.py)."""
    if df.empty:
        return []
    events = []
    for label in classes:
        found = RULES[label](df, scene, signals) if label in NEEDS_SIGNALS else RULES[label](df, scene)
        for s, e, why in merge(found):
            e = min(e, duration)
            if e > s:
                events.append((round(float(s), 2), round(float(e), 2), label, why))
    return sorted(events)
