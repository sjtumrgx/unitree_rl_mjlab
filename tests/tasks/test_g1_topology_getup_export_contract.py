from __future__ import annotations

from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_runner_cls

from src.tasks.velocity.rl.topology_getup_contract import (
  build_topology_getup_deploy_cfg,
  get_support_geometry_metadata,
)
from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner


def test_topology_getup_tasks_use_custom_runner() -> None:
  assert load_runner_cls("Unitree-G1-TopologyGetUp-Stage0") is TopologyGetupOnPolicyRunner
  assert load_runner_cls("Unitree-G1-TopologyGetUp-Benchmark") is TopologyGetupOnPolicyRunner


def test_topology_getup_support_geometry_metadata_is_versioned() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    metadata = get_support_geometry_metadata(env)
  finally:
    env.close()

  assert metadata["support_geometry_interface_version"] == "sgi_v1"
  assert metadata["support_geometry_patch_shape"] == [32, 32]
  assert metadata["support_geometry_anchor_names"] == [
    "trunk",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
  ]
  assert metadata["support_geometry_depth_camera_contract"]["sensor_name"] == "support_depth"
  assert metadata["support_geometry_depth_camera_contract"]["pointcloud_mode"] == "euclidean_norm"
  assert metadata["support_geometry_depth_camera_contract"]["timeout_ms"] == 500
  assert metadata["support_geometry_depth_camera_contract"]["retain_last_valid_frame"] is True
  assert metadata["support_geometry_depth_camera_contract"]["organized_pointcloud"] is True
  assert metadata["support_geometry_depth_camera_contract"]["pointcloud_field_names"] == {
    "x": "x",
    "y": "y",
    "z": "z",
  }
  assert metadata["support_geometry_student_obs_groups"]["camera"] == ["support_depth"]


def test_topology_getup_deploy_yaml_template_matches_sgi_contract() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    deploy_cfg = build_topology_getup_deploy_cfg(env)
  finally:
    env.close()

  support_iface = deploy_cfg["support_geometry_interface"]
  assert support_iface["version"] == "sgi_v1"
  assert support_iface["patch_shape"] == [32, 32]
  assert support_iface["missing_data_policy"] == "zeros"
  assert support_iface["depth_camera"]["sensor_name"] == "support_depth"
  assert support_iface["depth_camera"]["pointcloud_mode"] == "euclidean_norm"
  assert support_iface["depth_camera"]["timeout_ms"] == 500
  assert support_iface["depth_camera"]["retain_last_valid_frame"] is True
  assert support_iface["depth_camera"]["organized_pointcloud"] is True
  assert support_iface["depth_camera"]["pointcloud_field_names"] == {"x": "x", "y": "y", "z": "z"}
  assert "camera" in deploy_cfg["observations"]
  assert tuple(deploy_cfg["observations"]["camera"]) == ("support_depth",)
  assert deploy_cfg["observations"]["camera"]["support_depth"]["params"]["expected_size"] == 1024


def test_repo_deploy_yaml_template_has_matching_support_interface_fields() -> None:
  template_path = Path("deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml")
  payload = yaml.safe_load(template_path.read_text())
  assert payload["support_geometry_interface"]["version"] == "sgi_v1"
  assert payload["support_geometry_interface"]["depth_camera"]["sensor_name"] == "support_depth"
  assert payload["support_geometry_interface"]["depth_camera"]["pointcloud_mode"] == "euclidean_norm"
  assert payload["support_geometry_interface"]["depth_camera"]["timeout_ms"] == 500
  assert payload["support_geometry_interface"]["depth_camera"]["retain_last_valid_frame"] is True
  assert payload["support_geometry_interface"]["depth_camera"]["organized_pointcloud"] is True
  assert payload["support_geometry_interface"]["depth_camera"]["pointcloud_field_names"] == {
    "x": "x",
    "y": "y",
    "z": "z",
  }
  assert payload["observations"]["camera"]["support_depth"]["params"]["expected_size"] == 1024


def test_deploy_observation_registry_contains_support_depth_term() -> None:
  header = Path("deploy/include/isaaclab/envs/mdp/observations/observations.h").read_text()
  assert "REGISTER_OBSERVATION(support_depth)" in header
