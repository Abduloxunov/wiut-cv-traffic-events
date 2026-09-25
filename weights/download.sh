#!/usr/bin/env bash
# Fetch model weights once, with internet, before the offline evaluation run.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://github.com/ultralytics/assets/releases/download/v8.4.0
for w in yolo26m.pt; do
  [ -f "$w" ] || curl -L --fail -o "$w" "$BASE/$w"
done
echo "weights ready in $(pwd)"
