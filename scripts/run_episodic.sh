#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

models=()
while [[ $# -gt 0 && "$1" != --* ]]; do
  models+=("$1")
  shift
done
if [[ ${#models[@]} -eq 0 ]]; then
  echo "Usage: scripts/run_episodic.sh MODEL [MODEL ...] --output-dir DIR [runner options]" >&2
  exit 2
fi
args=()
for model in "${models[@]}"; do args+=(--model "$model"); done
python -m adapters.runner --mode episodic "${args[@]}" "$@"

