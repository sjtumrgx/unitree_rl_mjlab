from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import mjlab.tasks  # noqa: F401
import onnx
import src.tasks  # noqa: F401
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


def _metadata_map(path: Path) -> dict[str, str]:
  model = onnx.load(path)
  return {entry.key: entry.value for entry in model.metadata_props}


def test_onpolicy_export_artifacts_include_sgi_contract_and_policy_yaml(tmp_path: Path) -> None:
  runner = _build_runner("Unitree-G1-TopologyGetUp-Stage0", TopologyGetupOnPolicyRunner, tmp_path)
  try:
    model_path = tmp_path / "model_0.pt"
    runner.save(str(model_path))
    assert model_path.exists()
    onnx_path = tmp_path / "policy.onnx"
    deploy_yaml_path = tmp_path / "params" / "deploy.yaml"
    assert onnx_path.exists()
    assert deploy_yaml_path.exists()

    metadata = _metadata_map(onnx_path)
    assert metadata["support_geometry_interface_version"] == "sgi_v1"
    assert metadata["support_geometry_missing_data_policy"] == "zeros"
    assert "support_geometry_depth_camera_contract" in metadata

    deploy_yaml = yaml.safe_load(deploy_yaml_path.read_text())
    assert deploy_yaml["support_geometry_interface"]["version"] == "sgi_v1"
    assert deploy_yaml["support_geometry_interface"]["depth_camera"]["sensor_name"] == "support_depth"
    assert deploy_yaml["support_geometry_interface"]["depth_camera"]["pointcloud_mode"] == "euclidean_norm"
  finally:
    runner.env.close()


def test_teacher_export_writes_artifact_manifest_for_distillation_handoff(tmp_path: Path) -> None:
  runner = _build_runner("Unitree-G1-TopologyGetUp-Stage0-Teacher", TopologyGetupOnPolicyRunner, tmp_path)
  try:
    runner.save(str(tmp_path / "model_0.pt"))
    manifest_path = tmp_path / "topology_getup_artifacts.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == "topology_getup_artifacts_v1"
    assert payload["lane"] == "teacher"
    assert payload["checkpoint"] == "model_0.pt"
    assert payload["policy_onnx"] == "policy.onnx"
    assert "policy_analysis_onnx" not in payload
    assert payload["deploy_yaml"] == "params/deploy.yaml"
    assert payload["support_geometry_interface_version"] == "sgi_v1"
  finally:
    runner.env.close()


def test_naive_export_manifest_does_not_claim_missing_analysis_onnx(tmp_path: Path) -> None:
  runner = _build_runner("Unitree-G1-TopologyGetUp-Stage0-NaiveDepth", TopologyGetupOnPolicyRunner, tmp_path)
  try:
    runner.save(str(tmp_path / "model_0.pt"))
    manifest_path = tmp_path / "topology_getup_artifacts.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["lane"] == "naive_depth"
    assert payload["policy_onnx"] == "policy.onnx"
    assert "policy_analysis_onnx" not in payload
    assert not (tmp_path / "policy_analysis.onnx").exists()
  finally:
    runner.env.close()


def test_distillation_export_artifacts_mark_distillation_mode_and_preserve_sgi_contract(tmp_path: Path) -> None:
  runner = _build_runner(
    "Unitree-G1-TopologyGetUp-Stage0-Distill",
    TopologyGetupDistillationRunner,
    tmp_path,
  )
  try:
    import torch

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
    runner.save(str(tmp_path / "model_0.pt"))

    metadata = _metadata_map(tmp_path / "policy.onnx")
    assert metadata["distillation_mode"] == "teacher_student_topology_bottleneck"
    assert metadata["support_geometry_interface_version"] == "sgi_v1"

    deploy_yaml = yaml.safe_load((tmp_path / "params" / "deploy.yaml").read_text())
    assert deploy_yaml["support_geometry_interface"]["version"] == "sgi_v1"
    assert deploy_yaml["observations"]["camera"]["support_depth"]["params"]["expected_size"] == 1024

    manifest_path = tmp_path / "topology_getup_artifacts.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["lane"] == "distill"
    assert manifest["distillation_mode"] == "teacher_student_topology_bottleneck"
    assert manifest["teacher_checkpoint"] == str(teacher_checkpoint)
    assert manifest["deploy_yaml"] == "params/deploy.yaml"
  finally:
    runner.env.close()
