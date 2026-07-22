"""Filtering and validation rules for Sequential-EQA benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(frozen=True)
class FilterConfig:
    """Controls which source rows and scenes are retained."""

    minimum_sequence_length: int = 1
    supported_scenes: frozenset[str] = field(default_factory=frozenset)
    excluded_scenes: frozenset[str] = field(default_factory=frozenset)
    drop_duplicates: bool = True


def normalize_row(row: Mapping[str, object]) -> dict[str, str]:
    """Return a normalized benchmark row without changing question content."""
    return {
        "scene": str(row.get("scene", "")).strip(),
        "floor": str(row.get("floor", "0")).strip() or "0",
        "question": str(row.get("question", "")).strip(),
        "answer": str(row.get("answer", "")).strip(),
    }


def filter_rows(
    rows: Iterable[Mapping[str, object]], config: FilterConfig
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Apply row-level rules and return retained rows plus rejection counts."""
    kept: list[dict[str, str]] = []
    rejected = {"missing_required": 0, "unsupported_scene": 0, "excluded_scene": 0, "duplicate": 0}
    seen: set[tuple[str, str, str]] = set()

    for raw in rows:
        row = normalize_row(raw)
        if not row["scene"] or not row["question"] or not row["answer"]:
            rejected["missing_required"] += 1
            continue
        if config.supported_scenes and row["scene"] not in config.supported_scenes:
            rejected["unsupported_scene"] += 1
            continue
        if row["scene"] in config.excluded_scenes:
            rejected["excluded_scene"] += 1
            continue
        identity = (row["scene"], row["question"], row["answer"])
        if config.drop_duplicates and identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        kept.append(row)
    return kept, rejected

