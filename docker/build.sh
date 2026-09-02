#!/bin/sh
# Build the sm121 image chain v1 -> v9. Each stage FROMs the previous one, so a failure
# at stage N makes every later stage meaningless: stop there and say which one failed.
#
# Tag names: this script tags stages `radixark/vllm-glm53-flash:sm121-vN` (the local build
# chain the Dockerfiles' FROM lines expect). The images the README tells you to pull are
# published under `ghcr.io/tonyd2wild/vllm-glm53-flash:` tags. Same content, different
# namespace -- the radixark tags are build-local and never pushed.
for i in $(seq 1 9); do
  if ! docker build -f "Dockerfile.glm53-sm121-v$i" -t "radixark/vllm-glm53-flash:sm121-v$i" .; then
    echo "build.sh: FAILED at stage v$i (Dockerfile.glm53-sm121-v$i); stages v$i-v9 not built" >&2
    exit 1
  fi
done
