"""Assemble the Hugging Face Space (Docker) that hosts the website and the live demo.

  python tools/build_space.py --out ../space
  hf upload <user>/<space> ../space . --repo-type space

The Space contains: demo/ (server, demo pipeline, website incl. media built by tools/render_results.py,
make_site_data.py, make_site_media.py), src/, scene/ (zones + reference background), a Dockerfile that installs the
CPU build of PyTorch and fetches the YOLO26s / YOLO26n weights at build time.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE = """FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH YOLO_OFFLINE=1 PYTHONUNBUFFERED=1
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN mkdir -p weights && for w in yolo26s.pt yolo26n.pt; do \\
      curl -L --fail -o weights/$w https://github.com/ultralytics/assets/releases/download/v8.4.0/$w; done
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "demo/server.py", "--port", "7860"]
"""

README = """---
title: Traffic Event Detection
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: WIUT Hackathon 2026 CV track - website and live demo
---

Team website and live demo for the WIUT Hackathon 2026 computer-vision track: traffic events and accident risk from a
fixed intersection camera. Code: https://github.com/Abduloxunov/wiut-cv-traffic-events
"""

REQUIREMENTS = """numpy>=1.24
opencv-python-headless>=4.8
ultralytics==8.4.161
lap>=0.5
pandas>=2.0
imageio-ffmpeg>=0.5
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "space"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", out / "src", ignore=ignore)
    (out / "scene").mkdir()
    for f in ("zones.json", "background.jpg"):
        shutil.copy(ROOT / "scene" / f, out / "scene" / f)
    (out / "demo").mkdir()
    for f in ("server.py", "process.py"):
        shutil.copy(ROOT / "demo" / f, out / "demo" / f)
    shutil.copytree(ROOT / "demo" / "site", out / "demo" / "site", ignore=ignore)
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / ".gitattributes").write_text("*.mp4 filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"space built in {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
