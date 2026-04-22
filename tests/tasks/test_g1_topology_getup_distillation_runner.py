from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.tasks.velocity.rl.topology_getup_distillation_runner import TopologyGetupDistillationRunner


def _build_runner(tmp_path: Path) -> TopologyGetupDistillationRunner:
  env_cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0-Distill")
  env_cfg.scene.num_envs = 1
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
  agent_cfg = asdict(load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0-Distill"))
  agent_cfg["logger"] = "tensorboard"
  agent_cfg["upload_model"] = False
  return TopologyGetupDistillationRunner(env, agent_cfg, str(tmp_path), "cpu")


def test_distillation_runner_initializes_student_and_teacher_models(tmp_path: Path) -> None:
  runner = _build_runner(tmp_path)
  try:
    assert runner.alg.teacher_loaded is False
    assert runner.alg.student.__class__.__name__ == "TopologyBottleneckCNNModel"
    assert runner.alg.teacher.__class__.__name__ == "SpatialSoftmaxCNNModel"
  finally:
    runner.env.close()


def test_distillation_runner_can_load_teacher_checkpoint_and_export_student_contract(tmp_path: Path) -> None:
  runner = _build_runner(tmp_path)
  try:
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save(
      {
        "actor_state_dict": runner.alg.teacher.state_dict(),
        "iter": 0,
        "infos": {},
      },
      teacher_checkpoint,
    )
    runner._teacher_load_path = str(teacher_checkpoint)
    runner._maybe_load_teacher()
    assert runner.alg.teacher_loaded is True

    model_path = tmp_path / "model_0.pt"
    runner.save(str(model_path))
    assert model_path.exists()
    assert (tmp_path / "policy.onnx").exists()
    deploy_yaml = tmp_path / "params" / "deploy.yaml"
    assert deploy_yaml.exists()
    text = deploy_yaml.read_text()
    assert "support_geometry_interface" in text
    assert "support_depth" in text
  finally:
    runner.env.close()
