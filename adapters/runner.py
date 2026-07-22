#!/usr/bin/env python3
"""Run one or more adapters over canonical Sequential-EQA scene files."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .base import AdapterConfig, EvaluationMode, ResultRecord, write_jsonl
from .exploreeqa_adapter import ExploreEQAAdapter
from .memoryeqa_adapter import MemoryEQAAdapter
from .three_d_mem_adapter import ThreeDimensionalMemoryAdapter
from .uninavid_adapter import UniNaVidAdapter

ADAPTERS = {
    "exploreeqa": ExploreEQAAdapter,
    "memoryeqa": MemoryEQAAdapter,
    "3D-Mem": ThreeDimensionalMemoryAdapter,
    "uninavid": UniNaVidAdapter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in EvaluationMode])
    parser.add_argument("--model", action="append", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--sequences", type=Path, default=Path("benchmark/sequence_files/openeqa_hm3d"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene", action="append", help="Restrict to one or more scene IDs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = EvaluationMode(args.mode)
    scenes = set(args.scene or [])
    sequence_files = sorted(args.sequences.glob("*.csv"))
    if scenes:
        sequence_files = [path for path in sequence_files if path.stem in scenes]
    if not sequence_files:
        raise SystemExit(f"No matching sequence CSVs in {args.sequences}")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    for model in args.model:
        config = AdapterConfig.from_yaml(args.config_dir / f"{model}.yaml")
        adapter = ADAPTERS[model](config)
        all_records = []
        for sequence in sequence_files:
            native = args.output_dir / run_id / model / mode.value / "native" / sequence.stem
            command = adapter.run_inference(mode, sequence.resolve(), native.resolve(), dry_run=args.dry_run)
            if args.dry_run:
                print(shlex.join(command))
            else:
                worker_output = native / "results.jsonl"
                with worker_output.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
                            row["run_id"] = run_id
                            all_records.append(ResultRecord(**row))
        if not args.dry_run:
            output = args.output_dir / run_id / model / mode.value / "results.jsonl"
            write_jsonl(all_records, output)
            metadata = {"run_id": run_id, "model": model, "mode": mode.value, "questions": len(all_records)}
            output.with_name("run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
