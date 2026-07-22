#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT_DIR/external/explore-eqa"
URL="https://github.com/Stanford-ILIAD/explore-eqa.git"
if [[ ! -d "$DEST/.git" ]]; then git clone "$URL" "$DEST"; fi
git -C "$DEST" pull --ff-only
conda env create -n sequential-eqa-exploreeqa -f "$DEST/environment.yml" || conda env update -n sequential-eqa-exploreeqa -f "$DEST/environment.yml"
conda run -n sequential-eqa-exploreeqa pip install -e "$DEST"
echo "ExploreEQA installed at $(git -C "$DEST" rev-parse HEAD). The adapter imports it directly."
