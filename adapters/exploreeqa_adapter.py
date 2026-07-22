"""Import-based adapter for the current ExploreEQA repository."""

from __future__ import annotations

import csv
import pickle
import time
from pathlib import Path

from .base import BaseAdapter


class ExploreEQAAdapter(BaseAdapter):
    """Reuse the upstream VLM and TSDF map without modifying upstream files."""

    module = None
    cfg = None
    vlm = None
    planner = None
    _real_vlm = None
    _real_planner = None
    _preserve_memory = False

    def load_model(self) -> None:
        omega = self.import_upstream("omegaconf")
        self.module = self.import_upstream("run_vlm_exp")
        cfg_path = self.config.repository / str(
            self.config.parameters.get("native_config", "cfg/vlm_exp.yaml")
        )
        self.cfg = omega.OmegaConf.load(cfg_path)
        if self.config.parameters.get("hm3d_data"):
            self.cfg.scene_data_path = str(self.config.parameters["hm3d_data"])
        if self.config.parameters.get("initial_poses"):
            self.cfg.init_pose_data_path = str(self.config.parameters["initial_poses"])
        self.cfg.save_obs = True
        self._real_vlm = self.module.VLM
        self._real_planner = self.module.TSDFPlanner

        def vlm_factory(vlm_cfg):
            if self.vlm is None:
                self.vlm = self._real_vlm(vlm_cfg)
            return self.vlm

        def planner_factory(*args, **kwargs):
            if self.planner is None or not self._preserve_memory:
                self.planner = self._real_planner(*args, **kwargs)
            return self.planner

        self.module.VLM = vlm_factory
        self.module.TSDFPlanner = planner_factory

    def clear_memory(self) -> None:
        self.planner = None

    def reset_task_state(self) -> None:
        if self.planner is not None:
            for name in ("target_point", "target_direction", "max_point"):
                if hasattr(self.planner, name):
                    setattr(self.planner, name, None)

    def begin_question(self, question_id: str, preserve_memory: bool) -> None:
        self._preserve_memory = preserve_memory
        super().begin_question(question_id, preserve_memory)

    def infer_question(self, question: dict[str, str], question_index: int) -> dict[str, object]:
        if self.module is None or self.cfg is None:
            raise RuntimeError("ExploreEQA is not loaded")
        episode_dir = (self.config.output_root / "exploreeqa-native" / question["scene"] / question["question_id"].split(":")[-1]).resolve()
        episode_dir.mkdir(parents=True, exist_ok=True)
        question_csv = episode_dir / "question.csv"
        with question_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("scene", "floor", "question", "choices", "answer"))
            writer.writeheader()
            writer.writerow({
                "scene": question["scene"], "floor": question.get("floor", "0"),
                "question": question["question"], "choices": "['', '', '', '']",
                "answer": question["answer"],
            })
        cfg = self.cfg.copy()
        cfg.question_data_path = str(question_csv)
        cfg.output_dir = str(episode_dir)
        started = time.monotonic()
        self.module.main(cfg)
        elapsed = time.monotonic() - started
        with (episode_dir / "results.pkl").open("rb") as handle:
            result = pickle.load(handle)[-1]
        steps = sorted((key for key in result if key.startswith("step_")), key=lambda key: int(key.split("_")[-1]))
        best_step = max(steps, key=lambda key: float(result[key].get("smx_vlm_rel", [0])[0]))
        image_path = episode_dir / "0" / f"{best_step.split('_')[-1]}.png"
        image_module = self.import_upstream("PIL.Image")
        prediction = self.vlm.generate(
            f"Answer the question directly and concisely: {question['question']}",
            image_module.open(image_path).convert("RGB"),
        )
        points = [result[key]["pts"] for key in steps if "pts" in result[key]]
        distance = sum(float(self.module.np.linalg.norm(self.module.np.asarray(b) - self.module.np.asarray(a))) for a, b in zip(points, points[1:]))
        return {
            "prediction": prediction,
            "path_length_m": distance,
            "navigation_time_s": elapsed,
            "num_steps": len(steps),
            "metadata": {"best_observation_step": best_step},
        }
