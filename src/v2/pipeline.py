"""Event layer v2: run every emitted class on the tracks of one video -> [[start, end, label], ...].

Classes and why they are (not) emitted -- see docs/worklog.md for the numbers on the dev labels:
  emitted: jaywalking, stopped_vehicle, failure_to_yield, congestion, stop_line (needs the signal timeline),
           solid_line_crossing (weak, but present in every sample video), illegal_turn,
           red_light (strict: settled red; fires only on the one real runner in the four samples),
           wrong_way (strict: sustained, on the straight carriageways; silent on the samples, 20/20 on reversed tracks)
  not emitted: near_miss (no working signal yet),
           accident, illegal_u_turn, road_obstacle, fire_smoke (never seen in the samples; an absent class that we
           predict adds a zero to the macro average)
"""
import time

from v2.congestion import Congestion
from v2.failure_to_yield import FailureToYield
from v2.illegal_turn import IllegalTurn
from v2.jaywalking import Jaywalking
from v2.red_light import RedLight
from v2.wrong_way import WrongWay
from v2.segments import union
from v2.solid_line import SolidLine
from v2.stop_line import StopLine
from v2.stopped_vehicle import StoppedVehicle

EMITTED = ("jaywalking", "stopped_vehicle", "failure_to_yield", "congestion", "stop_line",
           "solid_line_crossing", "illegal_turn", "red_light", "wrong_way")


def detect_v2(df, scene, duration, signals=None, classes=EMITTED, fps=29.97, timings=None):
    """df: tracks with motion columns (rules.add_motion). Returns sorted [[start, end, label]] with same-class
    segments merged and everything clipped to [0, duration]."""
    runners = {
        "jaywalking": lambda: Jaywalking(scene).detect(df),
        "stopped_vehicle": lambda: StoppedVehicle(scene, fps=fps).detect(df, duration),
        "failure_to_yield": lambda: FailureToYield(scene).detect(df),
        "congestion": lambda: Congestion(scene).detect(df),
        "stop_line": lambda: StopLine(scene, signals).detect(df) if signals is not None and len(signals) else [],
        "solid_line_crossing": lambda: SolidLine(scene).detect(df),
        "illegal_turn": lambda: IllegalTurn(scene).detect(df),
        "red_light": lambda: RedLight(scene, signals).detect(df) if signals is not None and len(signals) else [],
        "wrong_way": lambda: WrongWay(scene).detect(df),
    }
    events = []
    for label in classes:
        if label not in runners:
            continue
        t0 = time.time()
        try:
            segs = [(s, e) for s, e, *_ in runners[label]()]
        except Exception as exc:  # one broken class must not take the whole video down
            print(f"  ! {label} failed: {exc!r}")
            segs = []
        if timings is not None:
            timings[label] = time.time() - t0
        for s, e in union(segs):
            s, e = max(0.0, float(s)), min(float(duration), float(e))
            if e - s > 1e-3:
                events.append([round(s, 3), round(e, 3), label])
    return sorted(events)
