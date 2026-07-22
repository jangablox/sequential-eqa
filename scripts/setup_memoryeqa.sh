#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/external/MemoryEQA"
URL="https://github.com/memory-eqa/MemoryEQA.git"
if [[ ! -d "$DEST/.git" ]]; then git clone "$URL" "$DEST"; fi
git -C "$DEST" pull --ff-only
conda create -n sequential-eqa-memoryeqa python=3.10 -y || true
conda run -n sequential-eqa-memoryeqa pip install -e "$DEST"
echo "MemoryEQA installed at $(git -C "$DEST" rev-parse HEAD). Set MEMORYEQA_CHECKPOINT before inference."
