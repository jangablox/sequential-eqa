#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/external/Uni-NaVid"
URL="https://github.com/jzhzhang/Uni-NaVid.git"
if [[ ! -d "$DEST/.git" ]]; then git clone "$URL" "$DEST"; fi
git -C "$DEST" pull --ff-only
conda create -n sequential-eqa-uninavid python=3.10 -y || true
conda run -n sequential-eqa-uninavid pip install -e "$DEST"
echo "Uni-NaVid installed at $(git -C "$DEST" rev-parse HEAD). Set UNINAVID_CHECKPOINT before inference."
