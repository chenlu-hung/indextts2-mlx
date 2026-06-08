#!/bin/bash
# Build the indextts2 CLI via xcodebuild so the MLX Metal shaders (default.metallib)
# get compiled. Plain `swift build` cannot compile the Metal kernels.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-Debug}"

xcodebuild \
    -scheme indextts2 \
    -destination 'platform=OS X' \
    -derivedDataPath .build/xcode \
    -configuration "$CONFIG" \
    build | tail -3

echo ""
echo "Binary: .build/xcode/Build/Products/$CONFIG/indextts2"
