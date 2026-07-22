#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/external/3D-Mem"
URL="https://github.com/UMass-Embodied-AGI/3D-Mem.git"
if [[ ! -d "$DEST/.git" ]]; then git clone "$URL" "$DEST"; fi
git -C "$DEST" pull --ff-only
conda create -n sequential-eqa-3d-mem python=3.9 -y || true
conda run -n sequential-eqa-3d-mem pip install -e "$DEST"
echo "3D-Mem installed at $(git -C "$DEST" rev-parse HEAD). Install its CUDA/Habitat dependencies from the upstream README."
