"""Common adapter contract and subprocess implementation helpers."""

from __future__ import annotations

import csv
import importlib
import json
import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import yaml


class EvaluationMode(str, Enum):
    EPISODIC = "episodic"
    SEQUENTIAL = "sequential"


@dataclass
class AdapterConfig:
    name: str
    repository: Path
    conda_env: str
    checkpoint: str = ""
    output_root: Path = Path("outputs")
    parameters: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "AdapterConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def expand(value: object) -> object:
            if isinstance(value, str):
                # Support shell-style ${NAME:-default} without invoking a shell.
                value = re.sub(
                    r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}",
                    lambda match: os.environ.get(match.group(1), match.group(2)),
                    value,
                )
                return os.path.expandvars(os.path.expanduser(value))
            if isinstance(value, dict):
                return {key: expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            return value

        raw = expand(raw)
        repository = Path(str(raw["repository"]))
        output_root = Path(str(raw.get("output_root", "outputs")))
        project_root = path.resolve().parent.parent
        if not repository.is_absolute():
            repository = project_root / repository
        if not output_root.is_absolute():
            output_root = project_root / output_root
        return cls(
            name=str(raw.get("name", path.stem)),
            repository=repository,
            conda_env=str(raw["conda_env"]),
            checkpoint=str(raw.get("checkpoint", "")),
            output_root=output_root,
            parameters=dict(raw.get("parameters", {})),
            source_path=path.resolve(),
        )


@dataclass
class ResultRecord:
    run_id: str
    model: str
    mode: str
    scene_id: str
    question_id: str
    question_index: int
    question: str
    reference_answer: str
    prediction: str = ""
    path_length_m: float | None = None
    navigation_time_s: float | None = None
    num_steps: int | None = None
    score: float | None = None
    status: str = "ok"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class BaseAdapter(ABC):
    """Adapter lifecycle shared by all model integrations.

    Heavy model code runs in its model-specific Conda environment. A scene is one
    process boundary: sequential mode retains permitted state within that process,
    while episodic mode clears it before every question.
    """

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config

    def validate_repository(self) -> None:
        if not self.config.repository.is_dir():
            raise FileNotFoundError(f"Repository not found: {self.config.repository}")

    def import_upstream(self, module_name: str):
        """Import an upstream module without copying it into this repository."""
        self.validate_repository()
        repository = str(self.config.repository.resolve())
        if repository not in sys.path:
            sys.path.insert(0, repository)
        return importlib.import_module(module_name)

    def load_model(self) -> None:
        """Import upstream code and initialize model resources in the worker."""

    def clear_memory(self) -> None:
        """Clear every model-specific persistent memory representation."""

    def reset_task_state(self) -> None:
        """Clear question-specific planning, prompting, and action state."""

    def reset_scene(self, scene_id: str) -> None:
        """Clear all persistent state at a scene boundary."""
        self.clear_memory()
        self.reset_task_state()

    def begin_question(self, question_id: str, preserve_memory: bool) -> None:
        """Reset task state and select whether scene memory may persist."""
        if not preserve_memory:
            self.clear_memory()
        self.reset_task_state()

    def build_command(
        self, mode: EvaluationMode, sequence_file: Path, native_output: Path
    ) -> list[str]:
        """Launch the import-based worker inside the model's Conda environment."""
        if self.config.source_path is None:
            raise ValueError("AdapterConfig.source_path is required")
        worker = Path(__file__).with_name("worker.py").resolve()
        return [
            "python", str(worker), "--model", self.config.name, "--config",
            str(self.config.source_path), "--mode", mode.value, "--sequence",
            str(sequence_file), "--output", str(native_output / "results.jsonl"),
        ]

    def run_inference(
        self,
        mode: EvaluationMode,
        sequence_file: Path,
        native_output: Path,
        *,
        dry_run: bool = False,
    ) -> list[str]:
        self.validate_repository()
        command = self.build_command(mode, sequence_file, native_output)
        wrapped = ["conda", "run", "--no-capture-output", "-n", self.config.conda_env, *command]
        if not dry_run:
            native_output.mkdir(parents=True, exist_ok=True)
            subprocess.run(wrapped, cwd=self.config.repository, check=True)
        return wrapped

    @abstractmethod
    def infer_question(self, question: dict[str, str], question_index: int) -> dict[str, object]:
        """Run one question using objects imported from the current upstream checkout."""

    def execute_scene(
        self, mode: EvaluationMode, sequence_file: Path, output: Path, run_id: str
    ) -> list[ResultRecord]:
        """Execute the common lifecycle in the model-specific worker process."""
        with sequence_file.open(newline="", encoding="utf-8-sig") as handle:
            questions = list(csv.DictReader(handle))
        if not questions:
            return []
        self.load_model()
        scene_id = questions[0]["scene"]
        self.reset_scene(scene_id)
        records: list[ResultRecord] = []
        for index, question in enumerate(questions, start=1):
            preserve = mode is EvaluationMode.SEQUENTIAL and index > 1
            self.begin_question(question["question_id"], preserve_memory=preserve)
            started = time.monotonic()
            try:
                native = self.infer_question(question, index)
                records.append(
                    ResultRecord(
                        run_id=run_id,
                        model=self.config.name,
                        mode=mode.value,
                        scene_id=scene_id,
                        question_id=question["question_id"],
                        question_index=index,
                        question=question["question"],
                        reference_answer=question["answer"],
                        prediction=str(native.get("prediction", "")),
                        path_length_m=_number(native.get("path_length_m")),
                        navigation_time_s=_number(native.get("navigation_time_s")) or (time.monotonic() - started),
                        num_steps=_integer(native.get("num_steps")),
                        status=str(native.get("status", "ok")),
                        error=str(native["error"]) if native.get("error") else None,
                        metadata=dict(native.get("metadata", {})),
                    )
                )
            except Exception as exc:
                records.append(
                    ResultRecord(
                        run_id=run_id, model=self.config.name, mode=mode.value,
                        scene_id=scene_id, question_id=question["question_id"],
                        question_index=index, question=question["question"],
                        reference_answer=question["answer"], status="error", error=str(exc),
                    )
                )
                if bool(self.config.parameters.get("fail_fast", True)):
                    write_jsonl(records, output)
                    raise
        write_jsonl(records, output)
        return records

    def normalize_results(
        self,
        native_output: Path,
        sequence_file: Path,
        mode: EvaluationMode,
        run_id: str,
    ) -> list[ResultRecord]:
        """Normalize common legacy CSV columns to the canonical result schema."""
        with sequence_file.open(newline="", encoding="utf-8-sig") as handle:
            questions = list(csv.DictReader(handle))

        result_rows: list[dict[str, str]] = []
        for candidate in sorted(native_output.rglob("*.csv")):
            with candidate.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                prediction_columns = {"answer_pred", "prediction", "final_prediction"}
                if reader.fieldnames and prediction_columns.intersection(reader.fieldnames):
                    result_rows.extend(reader)

        by_question: dict[str, deque[dict[str, str]]] = defaultdict(deque)
        for row in result_rows:
            by_question[str(row.get("question", "")).strip().lower()].append(row)
        records: list[ResultRecord] = []
        for index, question in enumerate(questions, start=1):
            matches = by_question.get(question["question"].strip().lower())
            native = matches.popleft() if matches else {}
            prediction = str(
                native.get("answer_pred", native.get("prediction", native.get("final_prediction", "")))
            )
            records.append(
                ResultRecord(
                    run_id=run_id,
                    model=self.config.name,
                    mode=mode.value,
                    scene_id=question["scene"],
                    question_id=question.get("question_id") or f"{question['scene']}:q{index:03d}",
                    question_index=index,
                    question=question["question"],
                    reference_answer=question["answer"],
                    prediction=prediction,
                    path_length_m=_number(native.get("distance", native.get("path_length"))),
                    navigation_time_s=_number(native.get("time", native.get("duration_seconds"))),
                    num_steps=_integer(native.get("steps", native.get("num_steps"))),
                    status="ok" if prediction else "missing",
                    error=None if prediction else "No native result row matched this question",
                    metadata={"native_columns": sorted(native)} if native else {},
                )
            )
        return records


def _number(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def write_jsonl(records: Iterable[ResultRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
