#!/usr/bin/env python3
"""Compute paired episodic and sequential EQA metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_records(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row.get("question_id", ""))
        if not key:
            raise ValueError("Every result row requires question_id")
        if key in indexed:
            raise ValueError(f"Duplicate question_id: {key}")
        indexed[key] = row
    return indexed


def _values(rows: Iterable[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _advantage(episodic: float, sequential: float) -> float | None:
    return ((episodic - sequential) / episodic * 100.0) if episodic else None


def summarize(ep_rows: list[dict[str, object]], seq_rows: list[dict[str, object]], threshold: float = 4) -> dict[str, object]:
    ep_scores, seq_scores = _values(ep_rows, "score"), _values(seq_rows, "score")
    if len(ep_scores) != len(ep_rows) or len(seq_scores) != len(seq_rows):
        raise ValueError("All paired records must have numeric scores")
    ep_open = mean((score - 1.0) / 4.0 for score in ep_scores) * 100.0
    seq_open = mean((score - 1.0) / 4.0 for score in seq_scores) * 100.0
    ep_binary = mean(score >= threshold for score in ep_scores) * 100.0
    seq_binary = mean(score >= threshold for score in seq_scores) * 100.0
    result: dict[str, object] = {
        "questions": len(ep_rows),
        "open_eqa_score_episodic": ep_open,
        "open_eqa_score_sequential": seq_open,
        "memory_advantage": seq_open - ep_open,
        "binary_success_episodic": ep_binary,
        "binary_success_sequential": seq_binary,
        "binary_memory_advantage": seq_binary - ep_binary,
    }
    for field, label in (
        ("path_length_m", "path"),
        ("navigation_time_s", "time"),
        ("num_steps", "step"),
    ):
        ep_values, seq_values = _values(ep_rows, field), _values(seq_rows, field)
        result[f"total_{label}_episodic"] = sum(ep_values)
        result[f"total_{label}_sequential"] = sum(seq_values)
        result[f"mean_{label}_episodic"] = mean(ep_values) if ep_values else None
        result[f"mean_{label}_sequential"] = mean(seq_values) if seq_values else None
        result[f"{label}_advantage"] = _advantage(sum(ep_values), sum(seq_values))
    return result


def compute(ep_rows: list[dict[str, object]], seq_rows: list[dict[str, object]], threshold: float = 4) -> tuple[dict[str, object], list[dict[str, object]]]:
    episodic, sequential = index_records(ep_rows), index_records(seq_rows)
    if set(episodic) != set(sequential):
        missing_seq = sorted(set(episodic) - set(sequential))
        missing_ep = sorted(set(sequential) - set(episodic))
        raise ValueError(f"Unmatched question IDs; missing sequential={missing_seq[:5]}, missing episodic={missing_ep[:5]}")
    keys = sorted(episodic)
    overall = summarize([episodic[key] for key in keys], [sequential[key] for key in keys], threshold)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        row = episodic[key]
        groups[("model", str(row.get("model", "unknown")))].append(key)
        groups[("scene", str(row.get("scene_id", "unknown")))].append(key)
        groups[("question_position", str(row.get("question_index", "unknown")))].append(key)
    breakdown = []
    for (group_type, group_value), group_keys in sorted(groups.items()):
        metrics = summarize([episodic[key] for key in group_keys], [sequential[key] for key in group_keys], threshold)
        breakdown.append({"group_type": group_type, "group_value": group_value, **metrics})
    return overall, breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodic", required=True, type=Path)
    parser.add_argument("--sequential", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--success-threshold", type=float, default=4)
    args = parser.parse_args()
    overall, breakdown = compute(read_jsonl(args.episodic), read_jsonl(args.sequential), args.success_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    if breakdown:
        with (args.output_dir / "metrics_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(breakdown[0]))
            writer.writeheader()
            writer.writerows(breakdown)
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
