"""Helpers shared by the v2 event modules."""
import numpy as np

RIDE_VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def riders(df, people=None):
    """Boolean array over `people` (default: person rows of df): feet inside a vehicle / bicycle box of the same frame
    (passengers, cyclists, moped riders). Vectorised per frame."""
    people = df[df.cls == "person"] if people is None else people
    veh = df[df.cls.isin(RIDE_VEHICLES)]
    out = np.zeros(len(people), bool)
    if people.empty or veh.empty:
        return out
    boxes = {f: g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in veh.groupby("frame")}
    pos = np.arange(len(people))
    for f, idx in people.groupby("frame").indices.items():
        b = boxes.get(f)
        if b is None:
            continue
        x = people.gx.to_numpy()[idx][:, None]
        y = people.gy.to_numpy()[idx][:, None]
        out[pos[idx]] = ((b[None, :, 0] <= x) & (x <= b[None, :, 2]) & (b[None, :, 1] <= y) & (y <= b[None, :, 3] + 10)).any(1)
    return out
