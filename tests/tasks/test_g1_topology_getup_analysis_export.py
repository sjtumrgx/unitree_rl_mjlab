from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import onnx
import src.tasks  # noqa: F401
import torch
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.tasks.velocity.rl.topology_getup_distillation_runner import TopologyGetupDistillationRunner
from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner


def _build_runner(task_id: str, runner_cls, tmp_path: Path):
  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 1
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
  agent_cfg = asdict(load_rl_cfg(task_id))
  agent_cfg["logger"] = "tensorboard"
  agent_cfg["upload_model"] = False
  return runner_cls(env, agent_cfg, str(tmp_path), "cpu")


def _output_names(onnx_path: Path) -> list[str]:
  model = onnx.load(onnx_path)
  return [output.name for output in model.graph.output]


def test_main_runner_exports_policy_analysis_onnx_with_topology_latent(tmp_path: Path) -> None:
  runner = _build_runner("Unitree-G1-TopologyGetUp-Stage0", TopologyGetupOnPolicyRunner, tmp_path)
  try:
    runner.save(str(tmp_path / "model_0.pt"))
    analysis_path = tmp_path / "policy_analysis.onnx"
    assert analysis_path.exists()
    assert _output_names(analysis_path) == ["actions", "topology_latent"]
  finally:
    runner.env.close()


def test_distill_runner_exports_policy_analysis_onnx_with_topology_latent(tmp_path: Path) -> None:
  runner = _build_runner("Unitree-G1-TopologyGetUp-Stage0-Distill", TopologyGetupDistillationRunner, tmp_path)
  try:
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": runner.alg.teacher.state_dict(), "iter": 0, "infos": {}}, teacher_checkpoint)
    runner._teacher_load_path = str(teacher_checkpoint)
    runner._maybe_load_teacher()
    runner.save(str(tmp_path / "model_0.pt"))
    analysis_path = tmp_path / "policy_analysis.onnx"
    assert analysis_path.exists()
    assert _output_names(analysis_path) == ["actions", "topology_latent"]
    deploy_yaml = yaml.safe_load((tmp_path / "params" / "deploy.yaml").read_text())
    assert deploy_yaml["support_geometry_interface"]["version"] == "sgi_v1"
  finally:
    runner.env.close()
