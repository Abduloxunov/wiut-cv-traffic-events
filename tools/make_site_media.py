"""Copy the rendered results into the website and cut example frames: one per detected class (from our predictions on
the sample videos) and the failure cases from our diagnosis figures.

  python tools/make_site_media.py --assets ../website_assets --figures ../figures --site demo/site
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2

EXAMPLES = {  # (video, class) picked where the event is clearly visible; time = event midpoint
    "jaywalking": "C3902", "stopped_vehicle": "C3902", "failure_to_yield": "C3902", "congestion": "C3905",
    "stop_line": "C3905", "solid_line_crossing": "C3897", "illegal_turn": "C3902",
}
FAILURES = {  # our diagnosis figures (crops of the sample videos with our boxes)
    "fail_moped_rider.jpg": "09_failure_to_yield/fp_cw1_C3905_0m13.7.jpg",
    "fail_head_box.jpg": "09_failure_to_yield/fp_cw1_C3905_1m48.8.jpg",
    "fail_kerb_waiting.jpg": "06_jaywalking_v2/c97_left.jpg",
    "fail_id_switch.jpg": "06_jaywalking_v2/c97_893.jpg",
    "fail_windscreen_people.jpg": "07_detection_check/C3902_0164.0.jpg",
    "stopped_final.jpg": "08_stopped_vehicle/v2_final_C3902.jpg",
    "solid_lines.jpg": "11_solid_line/solid_lines_drawn.jpg",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="../website_assets")
    ap.add_argument("--figures", default="../figures")
    ap.add_argument("--site", default="demo/site")
    a = ap.parse_args()
    assets, site = Path(a.assets), Path(a.site)
    (site / "videos").mkdir(parents=True, exist_ok=True)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "img" / "examples").mkdir(parents=True, exist_ok=True)
    for js in sorted(assets.glob("C*.json")):
        shutil.copy(js, site / "data" / js.name)
        shutil.copy(assets / f"{js.stem}_annotated.mp4", site / "videos" / f"{js.stem}.mp4")
    for label, v in EXAMPLES.items():
        res = json.load(open(assets / f"{v}.json"))
        ev = [e for e in res["events"] if e[2] == label]
        if not ev:
            continue
        s, e, _ = sorted(ev, key=lambda x: -(x[1] - x[0]))[len(ev) // 2 if label != "stopped_vehicle" else 0]
        cap = cv2.VideoCapture(str(assets / f"{v}_annotated.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((s + min(e, s + 4)) / 2 * res["fps"]))
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(site / "img" / "examples" / f"{label}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(label, v, round(s, 1), round(e, 1))
    # 12 s muted loop for the top of the page
    import subprocess
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-ss", "36", "-t", "12", "-i",
                    str(site / "videos" / "C3897.mp4"), "-vf", "scale=960:-2", "-an", "-c:v", "libx264", "-crf", "30",
                    "-preset", "veryfast", "-movflags", "+faststart", str(site / "videos" / "hero.mp4")], check=True)
    for name, src in FAILURES.items():
        p = Path(a.figures) / src
        if p.exists():
            im = cv2.imread(str(p))
            if im.shape[1] > 1400:
                im = cv2.resize(im, (1400, int(im.shape[0] * 1400 / im.shape[1])))
            cv2.imwrite(str(site / "img" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
