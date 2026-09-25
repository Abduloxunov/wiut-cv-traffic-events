#!/usr/bin/env bash
# Fetch model weights once, with internet, before the offline evaluation run.
set -euo pipefail
cd "$(dirname "$0")"
URL=https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
[ -f yolo26s.pt ] || curl -L --fail -o yolo26s.pt "$URL"
echo "weights ready in $(pwd)"
