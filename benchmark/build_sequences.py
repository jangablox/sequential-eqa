#!/usr/bin/env python3
"""Build deterministic scene-level Sequential-EQA question sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from .filters import FilterConfig, filter_rows
except ImportError:  # Allow direct execution from the repository root.
    from filters import FilterConfig, filter_rows

FIELDS = ("question_id", "scene", "floor", "question", "answer")


def discover_inputs(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if files:
            return files
    raise ValueError(f"Expected a CSV file or directory containing CSVs: {path}")


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"scene", "question", "answer"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} is missing required columns: {sorted(required)}")
            rows.extend(reader)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sequences(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
    shuffle: bool = False,
    minimum_sequence_length: int = 1,
    drop_duplicates: bool = True,
    source_dataset: str = "unknown",
    source_version: str = "unavailable",
) -> dict[str, object]:
    source_files = discover_inputs(input_path)
    rows, rejected = filter_rows(
        read_rows(source_files),
        FilterConfig(
            minimum_sequence_length=minimum_sequence_length,
            drop_duplicates=drop_duplicates,
        ),
    )
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene"]].append(row)

    grouped = {
        scene: scene_rows
        for scene, scene_rows in grouped.items()
        if len(scene_rows) >= minimum_sequence_length
    }
    if shuffle:
        rng = random.Random(seed)
        for scene in sorted(grouped):
            rng.shuffle(grouped[scene])

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    total = 0
    for scene in sorted(grouped):
        out_path = output_dir / f"{scene}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for index, row in enumerate(grouped[scene], start=1):
                writer.writerow({"question_id": f"{scene}:q{index:03d}", **row})
        count = len(grouped[scene])
        total += count
        files.append({"path": out_path.name, "scene_id": scene, "questions": count, "sha256": sha256(out_path)})

    manifest: dict[str, object] = {
        "benchmark": "Sequential-EQA Open-EQA HM3D",
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_dataset,
        "source_version": source_version,
        "ordering": "seeded_shuffle" if shuffle else "source_row_order",
        "seed": seed if shuffle else None,
        "settings": {"minimum_sequence_length": minimum_sequence_length, "drop_duplicates": drop_duplicates},
        "counts": {"scenes": len(files), "questions": total},
        "rejected": rejected,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--minimum-sequence-length", type=int, default=1)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--source-dataset", default="unknown")
    parser.add_argument("--source-version", default="unavailable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shuffle and args.seed is None:
        raise SystemExit("--shuffle requires --seed")
    manifest = build_sequences(
        args.input,
        args.output,
        seed=args.seed,
        shuffle=args.shuffle,
        minimum_sequence_length=args.minimum_sequence_length,
        drop_duplicates=not args.keep_duplicates,
        source_dataset=args.source_dataset,
        source_version=args.source_version,
    )
    counts = manifest["counts"]
    print(f"Wrote {counts['questions']} questions in {counts['scenes']} scene sequences to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
