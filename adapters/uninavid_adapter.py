"""Import-based adapter for the current Uni-NaVid repository."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from .base import BaseAdapter


class UniNaVidAdapter(BaseAdapter):
    """Retain Uni-NaVid's online feature cache only in sequential mode."""

    agent = None
    simulator = None
    habitat = None
    np = None
    initial_poses: dict[str, tuple[list[float], float]]

    def load_model(self) -> None:
        module = self.import_upstream("offline_eval_uninavid")
        self.habitat = self.import_upstream("habitat_sim")
        self.np = self.import_upstream("numpy")
        if not self.config.checkpoint or "$" in self.config.checkpoint:
            raise ValueError("Set UNINAVID_CHECKPOINT to the downloaded Uni-NaVid checkpoint")
        self.agent = module.UniNaVid_Agent(self.config.checkpoint)
        self.initial_poses = self._read_initial_poses(Path(str(self.config.parameters["initial_poses"])))

    @staticmethod
    def _read_initial_poses(path: Path) -> dict[str, tuple[list[float], float]]:
        poses = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = row.get("scene_floor", "")
                scene = key.rsplit("_", 1)[0] if key else row.get("scene", "")
                poses[scene] = (
                    [float(row[name]) for name in ("init_x", "init_y", "init_z")],
                    float(row.get("init_angle", 0)),
                )
        return poses

    def _scene_mesh(self, scene_id: str) -> Path:
        root = Path(str(self.config.parameters["hm3d_data"]))
        candidates = list(root.glob(f"*{scene_id}*/*.glb")) + list(root.glob(f"*{scene_id}*.glb"))
        if not candidates:
            raise FileNotFoundError(f"No HM3D mesh found for {scene_id} under {root}")
        basis = [path for path in candidates if ".basis" in path.name]
        return sorted(basis or candidates)[0]

    def _open_scene(self, scene_id: str) -> None:
        if self.simulator is not None:
            self.simulator.close()
        simulator_cfg = self.habitat.SimulatorConfiguration()
        simulator_cfg.scene_id = str(self._scene_mesh(scene_id))
        agent_cfg = self.habitat.agent.AgentConfiguration()
        sensor = self.habitat.CameraSensorSpec()
        sensor.uuid = "color_sensor"
        sensor.sensor_type = self.habitat.SensorType.COLOR
        sensor.resolution = [int(self.config.parameters.get("height", 480)), int(self.config.parameters.get("width", 640))]
        sensor.position = [0.0, float(self.config.parameters.get("camera_height", 1.5)), 0.0]
        sensor.hfov = float(self.config.parameters.get("hfov", 120))
        agent_cfg.sensor_specifications = [sensor]
        agent_cfg.action_space = {
            "move_forward": self.habitat.agent.ActionSpec("move_forward", self.habitat.agent.ActuationSpec(amount=float(self.config.parameters.get("forward_step", 0.25)))),
            "turn_left": self.habitat.agent.ActionSpec("turn_left", self.habitat.agent.ActuationSpec(amount=float(self.config.parameters.get("turn_angle", 30)))),
            "turn_right": self.habitat.agent.ActionSpec("turn_right", self.habitat.agent.ActuationSpec(amount=float(self.config.parameters.get("turn_angle", 30)))),
        }
        self.simulator = self.habitat.Simulator(self.habitat.Configuration(simulator_cfg, [agent_cfg]))

    def clear_memory(self) -> None:
        if self.agent is not None:
            self.agent.reset()

    def reset_task_state(self) -> None:
        if self.agent is not None:
            self.agent.pending_action_list = []
            self.agent.last_action = None
            self.agent.executed_steps = 0
            self.agent.rgb_list = []

    def reset_scene(self, scene_id: str) -> None:
        super().reset_scene(scene_id)
        self._open_scene(scene_id)

    def infer_question(self, question: dict[str, str], question_index: int) -> dict[str, object]:
        if self.agent is None or self.simulator is None:
            raise RuntimeError("Uni-NaVid is not loaded")
        position, angle = self.initial_poses[question["scene"]]
        sim_agent = self.simulator.get_agent(0)
        state = sim_agent.get_state()
        state.position = self.np.asarray(position)
        quat = self.import_upstream("habitat_sim.utils.common").quat_from_angle_axis(angle, self.np.asarray([0, 1, 0]))
        state.rotation = quat
        sim_agent.set_state(state)
        previous = self.np.asarray(position)
        distance = 0.0
        prediction = ""
        started = time.monotonic()
        max_steps = int(self.config.parameters.get("max_episode_steps", 2000))
        steps = 0
        stopped = False
        while steps < max_steps and not stopped:
            rgb = self.simulator.get_sensor_observations()["color_sensor"][..., :3]
            action_result = self.agent.act({"observations": rgb, "instruction": question["question"]})
            actions = action_result.get("actions", self.agent.latest_action.get("actions", []))
            for action in actions:
                if action == "stop":
                    stopped = True
                    break
                habitat_action = {"forward": "move_forward", "left": "turn_left", "right": "turn_right"}.get(action)
                if habitat_action:
                    self.simulator.step(habitat_action)
                    current = self.np.asarray(sim_agent.get_state().position)
                    distance += float(self.np.linalg.norm(current - previous))
                    previous = current
                steps += 1
                if steps >= max_steps:
                    break
        rgb = self.simulator.get_sensor_observations()["color_sensor"][..., :3]
        self.agent.rgb_list.append(rgb)
        prediction = self.agent.predict_inference(
            f"Answer this question directly based on the visual history: {question['question']}"
        )
        return {
            "prediction": prediction,
            "path_length_m": distance,
            "navigation_time_s": time.monotonic() - started,
            "num_steps": steps,
        }
