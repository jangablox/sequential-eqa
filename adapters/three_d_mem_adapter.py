"""Import-based adapter for the current 3D-Mem repository."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from .base import BaseAdapter


class ThreeDimensionalMemoryAdapter(BaseAdapter):
    """Retain the upstream Scene reconstruction and TSDF in sequential mode."""

    module = None
    cfg = None
    scene = None
    planner = None
    _real_scene = None
    _real_planner = None
    _preserve_memory = False
    initial_poses: dict[str, tuple[list[float], float]]

    def load_model(self) -> None:
        omega = self.import_upstream("omegaconf")
        self.module = self.import_upstream("run_aeqa_evaluation")
        cfg_path = self.config.repository / str(
            self.config.parameters.get("native_config", "cfg/eval_aeqa.yaml")
        )
        self.cfg = omega.OmegaConf.load(cfg_path)
        if self.config.parameters.get("hm3d_data"):
            self.cfg.scene_data_path = str(self.config.parameters["hm3d_data"])
        self.cfg.save_visualization = bool(self.config.parameters.get("save_visualization", False))
        self._real_scene = self.module.Scene
        self._real_planner = self.module.TSDFPlanner
        self.initial_poses = self._read_initial_poses(Path(str(self.config.parameters["initial_poses"])))

        def scene_factory(*args, **kwargs):
            if self.scene is None or not self._preserve_memory:
                self.scene = self._real_scene(*args, **kwargs)
            return self.scene

        def planner_factory(*args, **kwargs):
            if self.planner is None or not self._preserve_memory:
                self.planner = self._real_planner(*args, **kwargs)
            return self.planner

        self.module.Scene = scene_factory
        self.module.TSDFPlanner = planner_factory

    @staticmethod
    def _read_initial_poses(path: Path) -> dict[str, tuple[list[float], float]]:
        poses = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = row.get("scene_floor", "")
                scene = key.rsplit("_", 1)[0] if key else row.get("scene", "")
                position = [float(row[name]) for name in ("init_x", "init_y", "init_z")]
                poses[scene] = (position, float(row.get("init_angle", 0)))
        return poses

    def clear_memory(self) -> None:
        self.scene = None
        self.planner = None

    def reset_task_state(self) -> None:
        if self.planner is not None:
            for name in ("max_point", "target_point", "target_direction"):
                if hasattr(self.planner, name):
                    setattr(self.planner, name, None)

    def begin_question(self, question_id: str, preserve_memory: bool) -> None:
        self._preserve_memory = preserve_memory
        super().begin_question(question_id, preserve_memory)

    def infer_question(self, question: dict[str, str], question_index: int) -> dict[str, object]:
        if self.module is None or self.cfg is None:
            raise RuntimeError("3D-Mem is not loaded")
        if question["scene"] not in self.initial_poses:
            raise KeyError(f"No initial pose for {question['scene']}")
        position, angle = self.initial_poses[question["scene"]]
        rotation = [math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0]
        episode_dir = (self.config.output_root / "3D-Mem-native" / question["scene"] / question["question_id"].split(":")[-1]).resolve()
        episode_dir.mkdir(parents=True, exist_ok=True)
        questions_path = episode_dir / "questions.json"
        questions_path.write_text(json.dumps([{
            "question": question["question"], "answer": question["answer"],
            "question_id": question["question_id"], "episode_history": question["scene"],
            "position": position, "rotation": rotation,
        }], indent=2) + "\n", encoding="utf-8")
        cfg = self.cfg.copy()
        cfg.questions_list_path = str(questions_path)
        cfg.output_dir = str(episode_dir)
        started = time.monotonic()
        self.module.main(cfg)
        elapsed = time.monotonic() - started
        answers = json.loads((episode_dir / "gpt_answer.json").read_text())
        answer_entry = next((item for item in answers if item.get("question_id") == question["question_id"]), {})
        path_lengths = json.loads((episode_dir / "path_length.json").read_text())
        frames_path = episode_dir / "n_total_frames.json"
        frames = json.loads(frames_path.read_text()) if frames_path.exists() else {}
        return {
            "prediction": answer_entry.get("answer", ""),
            "path_length_m": path_lengths.get(question["question_id"]),
            "navigation_time_s": elapsed,
            "num_steps": frames.get(question["question_id"]),
        }
