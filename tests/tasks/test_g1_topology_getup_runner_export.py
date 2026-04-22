from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import onnx
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.tasks.velocity.rl.topology_getup_distillation_runner import TopologyGetupDistillationRunner
from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner


def _metadata_map(onnx_path: Path) -> dict[str, str]:
  model = onnx.load(onnx_path)
  return {entry.key: entry.value for entry in model.metadata_props}


def _build_runner(task_id: str, runner_cls, tmp_path: Path):
  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 1
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
  agent_cfg = asdict(load_rl_cfg(task_id))
  agent_cfg["logger"] = "tensorboard"
  agent_cfg["upload_model"] = False
  return runner_cls(env, agent_cfg, str(tmp_path), "cpu")


def test_topology_getup_onpolicy_runner_save_exports_sgi_metadata_and_deploy_yaml(tmp_path: Path) -> None:
  runner = _build_runner(
    "Unitree-G1-TopologyGetUp-Stage0",
    TopologyGetupOnPolicyRunner,
    tmp_path,
  )
  try:
    model_path = tmp_path / "model_0.pt"
    runner.save(str(model_path))
    assert model_path.exists()
    onnx_path = tmp_path / "policy.onnx"
    assert onnx_path.exists()
    metadata = _metadata_map(onnx_path)
    assert metadata["support_geometry_interface_version"] == "sgi_v1"
    assert metadata["support_geometry_anchor_names"] == "trunk,left_hand,right_hand,left_foot,right_foot"
    assert metadata["support_geometry_depth_camera_contract"]
    deploy_yaml = tmp_path / "params" / "deploy.yaml"
    assert deploy_yaml.exists()
    deploy_text = deploy_yaml.read_text()
    assert "support_geometry_interface" in deploy_text
    assert "support_depth" in deploy_text
  finally:
    runner.env.close()


def test_topology_getup_distillation_runner_save_marks_distillation_mode_in_onnx(tmp_path: Path) -> None:
  runner = _build_runner(
    "Unitree-G1-TopologyGetUp-Stage0-Distill",
    TopologyGetupDistillationRunner,
    tmp_path,
  )
  try:
    teacher_checkpoint = tmp_path / "teacher.pt"
    import torch

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
    runner.save(str(tmp_path / "model_0.pt"))
    metadata = _metadata_map(tmp_path / "policy.onnx")
    assert metadata["distillation_mode"] == "teacher_student_topology_bottleneck"
    assert metadata["support_geometry_interface_version"] == "sgi_v1"
  finally:
    runner.env.close()
