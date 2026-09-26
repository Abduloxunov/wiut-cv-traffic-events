"""Build one zip per video for teammates to label: the proxy video, the label tool with that video's proposals
built in, and a start script. Nothing to install: the page runs from the unzipped folder in any browser.

  python tools/make_label_bundles.py --proxies ../proxies --out ../label_bundles

Footage stays inside the team (AI Lab data condition): send the zips only to team members.
"""
import argparse
import json
import zipfile
from pathlib import Path

import cv2

TOOL = Path(__file__).with_name("label_tool.html")

START_BAT = '@echo off\r\nstart "" "%~dp0label_tool.html"\r\n'
START_SH = """#!/bin/sh
cd "$(dirname "$0")"
if command -v xdg-open >/dev/null; then xdg-open label_tool.html; else open label_tool.html; fi
"""
README = """Labelling {video}  ({minutes:.1f} min)

1. Unzip, then double-click START.bat (Windows) or run: sh start.sh (Mac/Linux).
   It opens label_tool.html in your browser (Chrome or Edge), with the video and {n} suggestions loaded.
   Keep the video next to label_tool.html.
2. Pass 1, suggestions (dashed): press N to jump to the next one and watch it.
   A = accept, [ / ] = move start/end to the current frame, Del = delete, Shift+class key = change class.
3. Pass 2: watch the whole video at 2x ("." faster, "," slower). For anything missed:
   press the class key, I at the start frame, O at the end frame. Left/Right = 1 frame.
4. Ctrl+Z undoes. Work is saved in this browser automatically, but press "Export labels.json" often:
   it downloads labels_{stem}.json. Send that file back to Davlatyor.

Rules
- The panel on the right shows the definition of the selected class: start and end follow it exactly.
- One segment = one event of one class. Two jaywalkers at the same time = one segment covering both.
- Not sure? Label it anyway and write the time and your doubt in notes.txt, send it with the labels.
- Suggestions come from our rules and are often wrong: judge each one, don't accept by default.
"""


def duration(video):
    cap = cv2.VideoCapture(str(video))
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 29.97)
    cap.release()
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proxies", default="../proxies")
    p.add_argument("--out", default="../label_bundles")
    a = p.parse_args()

    proxies, out = Path(a.proxies), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tool = TOOL.read_text(encoding="utf-8")
    assert tool.count("const PRELOAD = null;") == 1
    for video in sorted(v for v in proxies.iterdir() if v.suffix.lower() == ".mp4"):
        props_path = proxies / f"proposals_{video.stem}.json"
        props = json.loads(props_path.read_text()) if props_path.exists() else {"videos": {}}
        entry = props["videos"].get(video.name, {})
        preload = {"video": video.name, "proposals": {"videos": {video.name: entry}}}
        page = tool.replace("const PRELOAD = null;", "const PRELOAD = " + json.dumps(preload) + ";")
        readme = README.format(video=video.name, stem=video.stem, n=len(entry.get("proposals", [])),
                               minutes=duration(video) / 60)
        zpath = out / f"label_{video.stem}.zip"
        folder = f"label_{video.stem}/"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(folder + "label_tool.html", page)
            z.writestr(folder + "README.txt", readme)
            z.writestr(folder + "START.bat", START_BAT)
            info = zipfile.ZipInfo(folder + "start.sh")
            info.external_attr = 0o755 << 16
            z.writestr(info, START_SH)
            z.write(video, folder + video.name, compress_type=zipfile.ZIP_STORED)  # H.264 does not compress
        print(f"{zpath}  {zpath.stat().st_size / 1e6:.0f} MB  ({len(entry.get('proposals', []))} proposals)")


if __name__ == "__main__":
    main()
