"""Write the solution diagram as an Excalidraw file (open at excalidraw.com or with the VS Code Excalidraw extension).

  python tools/make_pipeline_excalidraw.py --out docs/pipeline.excalidraw
"""
import argparse
import json
import random
from pathlib import Path

COL = {  # background colours (Excalidraw palette)
    "learned": "#a5d8ff", "rule": "#e9ecef", "data": "#fff3bf", "out": "#b2f2bb", "dev": "#d0bfff", "off": "#ffc9c9",
}
els = []
rnd = random.Random(7)


def _base(kind, x, y, w, h, **kw):
    e = {"id": f"{kind}-{len(els)}", "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
         "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "solid", "strokeWidth": 2,
         "strokeStyle": "solid", "roughness": 1, "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
         "seed": rnd.randint(1, 2**31), "version": 1, "versionNonce": rnd.randint(1, 2**31), "isDeleted": False,
         "boundElements": [], "updated": 1, "link": None, "locked": False}
    e.update(kw)
    els.append(e)
    return e


def text(x, y, s, size=16, w=None, container=None, align="center", color="#1e1e1e"):
    lines = s.split("\n")
    w = w or max(len(l) for l in lines) * size * 0.55
    h = len(lines) * size * 1.25
    return _base("text", x, y, w, h, text=s, originalText=s, fontSize=size, fontFamily=1, textAlign=align,
                 verticalAlign="middle", containerId=container, autoResize=True, lineHeight=1.25, strokeColor=color)


def box(x, y, w, h, label, kind="rule", size=16, dashed=False):
    r = _base("rectangle", x, y, w, h, backgroundColor=COL[kind], roundness={"type": 3},
              strokeStyle="dashed" if dashed else "solid")
    t = text(x + 8, y + h / 2 - label.count("\n") * size * 0.63 - size * 0.63, label, size, w - 16, r["id"])
    r["boundElements"].append({"type": "text", "id": t["id"]})
    return r


def arrow(a, b, label=None, via=None):
    """Arrow from the right edge of a to the left edge of b (or given absolute points)."""
    if via is None:
        x0, y0 = a["x"] + a["width"], a["y"] + a["height"] / 2
        x1, y1 = b["x"], b["y"] + b["height"] / 2
        pts = [[0, 0], [x1 - x0, y1 - y0]]
    else:
        x0, y0 = via[0]
        pts = [[px - x0, py - y0] for px, py in via]
    ar = _base("arrow", x0, y0, abs(pts[-1][0]) or 1, abs(pts[-1][1]) or 1, points=pts, lastCommittedPoint=None,
               startBinding={"elementId": a["id"], "focus": 0, "gap": 4},
               endBinding={"elementId": b["id"], "focus": 0, "gap": 4},
               startArrowhead=None, endArrowhead="arrow", roundness={"type": 2}, strokeColor="#495057")
    a["boundElements"].append({"type": "arrow", "id": ar["id"]})
    b["boundElements"].append({"type": "arrow", "id": ar["id"]})
    if label:
        mx, my = x0 + pts[-1][0] / 2, y0 + pts[-1][1] / 2
        text(mx - 60, my - 26, label, 13, 120, color="#495057")
    return ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/pipeline.excalidraw")
    a = ap.parse_args()

    text(40, 10, "Traffic event detection: how our system works", 28, 760, align="left")
    text(40, 52, "WIUT Hackathon 2026, CV track. Blue = learned, grey = rules / classical CV, yellow = data we made, "
                 "green = outputs", 16, 1100, align="left", color="#495057")

    # Part A main line
    text(40, 110, "Part A: events as [start, end, label]", 20, 420, align="left")
    video = box(40, 150, 170, 90, "4K video\n29.97 fps", "data")
    samp = box(250, 150, 170, 90, "Frame sampler\nevery 3rd frame\n(10 per second)")
    det = box(460, 150, 190, 90, "YOLO26m @1280\nCOCO-pretrained\nperson, car, bus...", "learned")
    trk = box(690, 150, 180, 90, "ByteTrack (tuned)\ntrack IDs")
    mot = box(910, 150, 190, 90, "Tracks + motion\nground point, speed,\nclass vote")
    ev = box(1140, 130, 260, 210, "Event layer v2\n(one module per class)\n\njaywalking\nstopped_vehicle\n"
             "failure_to_yield\ncongestion, stop_line\nsolid_line_crossing\nillegal_turn", "rule", 15)
    post = box(1440, 150, 200, 90, "Segment clean-up\nk-of-n confirm, gap\nbridging, merge, clip")
    outA = box(1680, 150, 180, 90, "[start, end, label]\nper video", "out")
    for x, y in [(video, samp), (samp, det), (det, trk), (trk, mot), (mot, ev), (ev, post), (post, outA)]:
        arrow(x, y)

    # Scene alignment branch
    first = box(250, 300, 170, 80, "First frame")
    sift = box(460, 300, 190, 80, "SIFT + RANSAC\nhomography")
    zones = box(690, 280, 220, 120, "Our map of the junction\ncrossings, lanes + directions,\nstop line, dividers,\n"
                "islands, signal heads", "data", 14)
    aligned = box(950, 300, 150, 80, "Zones in this\nvideo's pixels")
    arrow(video, first, via=[(125, 240), (125, 340), (250, 340)])
    arrow(first, sift)
    arrow(sift, zones)
    arrow(zones, aligned)
    arrow(aligned, ev, via=[(1100, 340), (1140, 300)])

    # Signal branch
    crops = box(460, 430, 190, 80, "Signal-head crops\n(signal_4, signal_5)")
    lamp = box(690, 430, 220, 80, "Lamp colour + lamp\nposition, flicker < 3 s off")
    lights = box(950, 430, 150, 80, "Light timeline\nred / green")
    arrow(samp, crops, via=[(335, 240), (335, 470), (460, 470)])
    arrow(crops, lamp)
    arrow(lamp, lights)
    arrow(lights, ev, via=[(1100, 470), (1180, 340)])

    # Not emitted
    box(1440, 290, 420, 110, "Not emitted (no reliable detector on the samples):\nnear_miss, red_light, wrong_way, "
        "accident,\nillegal_u_turn, road_obstacle, fire_smoke\n(an absent class we predict adds a zero)", "off", 14,
        dashed=True)

    # Part B
    text(40, 570, "Part B: causal P(accident starts within 5 s), one value per frame", 20, 700, align="left")
    frames = box(40, 610, 170, 90, "Frames in order\n(only the past)", "data")
    detB = box(250, 610, 190, 90, "YOLO26s @960\n+ ByteTrack\nevery 3rd frame", "learned")
    pairs = box(480, 610, 230, 90, "Every pair of road users:\nclosing speed, time to\ncontact (sizes/s)")
    drac = box(750, 610, 220, 90, "DRAC = deceleration\nneeded to avoid the crash\npersistence + smoothing")
    calib = box(1010, 610, 200, 90, "Calibrated: normal\ntraffic < 0.5\n(0 alarms in 7.4 min)")
    outB = box(1250, 610, 180, 90, "risk in [0, 1]\nper frame", "out")
    for x, y in [(frames, detB), (detB, pairs), (pairs, drac), (drac, calib), (calib, outB)]:
        arrow(x, y)

    # Dev loop
    text(40, 760, "How we tuned it (development loop, not part of the submission run)", 20, 760, align="left")
    tool = box(40, 800, 200, 90, "Our labelling tool\n+ per-video bundles", "dev")
    labels = box(280, 800, 200, 90, "Dev labels of the\nsample videos\n(58+ events)", "dev")
    evalu = box(520, 800, 220, 90, "Official evaluate.py\nper class, tIoU\n0.3 / 0.5 / 0.7", "dev")
    lovo = box(780, 800, 240, 90, "Settings chosen with\nleave-one-video-out;\nsensitivity checks", "dev")
    res = box(1060, 800, 240, 90, "Score A on our labels\nfirst rules 0.14\n-> event layer v2 0.49", "out")
    for x, y in [(tool, labels), (labels, evalu), (evalu, lovo), (lovo, res)]:
        arrow(x, y)

    doc = {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": els,
           "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"}, "files": {}}
    Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", a.out, len(els), "elements")


if __name__ == "__main__":
    main()
