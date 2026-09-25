#!/usr/bin/env bash
# One-time setup on a Linux machine with an NVIDIA GPU. Run from the project folder: bash setup_lab.sh
# Needs Python 3.10+ and internet for the package downloads.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision            # Linux wheels from PyPI include CUDA
pip install -r requirements.txt -r train/requirements-train.txt
[ -f weights/yolo26m.pt ] || bash weights/download.sh
python train/check_gpu.py
echo "Setup done. Next time just run: source .venv/bin/activate"
