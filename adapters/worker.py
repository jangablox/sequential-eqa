#!/usr/bin/env python3
"""In-environment worker that imports and executes one upstream model adapter."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.base import AdapterConfig, EvaluationMode
from adapters.exploreeqa_adapter import ExploreEQAAdapter
from adapters.memoryeqa_adapter import MemoryEQAAdapter
from adapters.three_d_mem_adapter import ThreeDimensionalMemoryAdapter
from adapters.uninavid_adapter import UniNaVidAdapter

ADAPTERS = {
    "exploreeqa": ExploreEQAAdapter,
    "memoryeqa": MemoryEQAAdapter,
    "3D-Mem": ThreeDimensionalMemoryAdapter,
    "uninavid": UniNaVidAdapter,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in EvaluationMode])
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = AdapterConfig.from_yaml(args.config)
    adapter = ADAPTERS[args.model](config)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    adapter.execute_scene(EvaluationMode(args.mode), args.sequence, args.output, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
