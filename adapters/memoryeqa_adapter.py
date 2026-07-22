"""Import-based adapter for the current MemoryEQA repository."""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseAdapter


class MemoryEQAAdapter(BaseAdapter):
    """Retain StructuredMemory only when the sequential protocol permits it."""

    model = None
    _preserve_memory = False

    def load_model(self) -> None:
        omega = self.import_upstream("omegaconf")
        module = self.import_upstream("src.modeling.memory_eqa")
        cfg_path = self.config.repository / str(
            self.config.parameters.get("native_config", "cfg/vlm_exp_ov.yaml")
        )
        cfg = omega.OmegaConf.load(cfg_path)
        if self.config.checkpoint and "$" not in self.config.checkpoint:
            cfg.vlm.model_name_or_path = self.config.checkpoint
        if self.config.parameters.get("hm3d_data"):
            cfg.scene_data_path = [str(self.config.parameters["hm3d_data"])]
        if self.config.parameters.get("initial_poses"):
            cfg.init_pose_data_path = str(self.config.parameters["initial_poses"])
        cfg.output_dir = str(self.config.output_root / "memoryeqa-native")
        self.model = module.MemoryEQA(cfg, int(self.config.parameters.get("gpu", 0)))

        original_prepare = self.model.prepare_data

        def prepare_open_vocab(question_data, question_ind):
            knowledge = getattr(self.model, "knowledge_base", None)
            original_clear = getattr(knowledge, "clear", None)
            if self._preserve_memory and original_clear is not None:
                knowledge.clear = lambda: None
            try:
                prepared = original_prepare(question_data, question_ind)
            finally:
                if self._preserve_memory and original_clear is not None:
                    knowledge.clear = original_clear
            meta = prepared[0]
            meta["question"] = re.sub(r"\n[A-D]\.\s*(?=\n|$)", "", meta["question"])
            return (meta, *prepared[1:])

        self.model.prepare_data = prepare_open_vocab

    def clear_memory(self) -> None:
        if self.model is not None and hasattr(self.model, "knowledge_base"):
            self.model.knowledge_base.clear()

    def reset_task_state(self) -> None:
        # MemoryEQA creates a fresh simulator and TSDF planner in prepare_data().
        pass

    def begin_question(self, question_id: str, preserve_memory: bool) -> None:
        self._preserve_memory = preserve_memory
        super().begin_question(question_id, preserve_memory)

    def infer_question(self, question: dict[str, str], question_index: int) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("MemoryEQA model is not loaded")
        row = dict(question)
        row.setdefault("floor", "0")
        row.setdefault("choices", "[]")
        result = self.model.run(row, question_index - 1)
        summary = result.get("summary", {})
        return {
            "prediction": summary.get("smx_vlm_pred", ""),
            "path_length_m": summary.get("path_length"),
            "navigation_time_s": summary.get("all_time_comsume"),
            "num_steps": len(result.get("step", [])),
            "metadata": {"upstream_result_keys": sorted(result)},
        }
