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


RULES = {
    "stopped_vehicle": stopped_vehicle,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "congestion": congestion,
    "wrong_way": wrong_way,
}


def detect(df, scene, duration, classes=tuple(RULES)):
    """Run the enabled rules -> [(start, end, label, why)], merged per class, clipped to the video."""
    if df.empty:
        return []
    events = []
    for label in classes:
        for s, e, why in merge(RULES[label](df, scene)):
            e = min(e, duration)
            if e > s:
                events.append((round(float(s), 2), round(float(e), 2), label, why))
    return sorted(events)
