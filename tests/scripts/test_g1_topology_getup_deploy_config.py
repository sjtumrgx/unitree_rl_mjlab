from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "deploy" / "robots" / "g1" / "config" / "config.yaml"
GETUP_CONFIG_PATH = ROOT / "deploy" / "robots" / "g1_getup" / "config" / "config.yaml"
DEPLOY_YAML_PATH = (
  ROOT / "deploy" / "robots" / "g1_getup" / "config" / "policy" / "topology_getup" / "v0" / "params" / "deploy.yaml"
)


def test_g1_deploy_config_remains_velocity_only_after_topology_getup_split() -> None:
  payload = yaml.safe_load(CONFIG_PATH.read_text())
  fsm = payload["FSM"]
  assert "Velocity" in fsm["_"]
  assert "TopologyGetUp" not in fsm["_"]
  assert fsm["Velocity"]["policy_dir"] == "config/policy/velocity"


def test_g1_getup_deploy_config_owns_topology_getup_runtime_entrypoint() -> None:
  payload = yaml.safe_load(GETUP_CONFIG_PATH.read_text())
  fsm = payload["FSM"]
  assert "TopologyGetUp" in fsm["_"]
  assert fsm["_"]["TopologyGetUp"]["type"] == "RLBase"
  assert fsm["FixStand"]["keyboard_transitions"]["TopologyGetUp"] == "g.on_pressed"
  assert fsm["FixStand"]["transitions"]["TopologyGetUp"] == "RT + Y.on_pressed"
  assert fsm["TopologyGetUp"]["policy_dir"] == "config/policy/topology_getup/v0"


def test_topology_getup_g1_getup_deploy_yaml_declares_support_geometry_interface() -> None:
  payload = yaml.safe_load(DEPLOY_YAML_PATH.read_text())
  assert payload["support_geometry_interface"]["version"] == "sgi_v1"
  assert payload["support_geometry_interface"]["depth_camera"]["sensor_name"] == "support_depth"
  assert payload["support_geometry_interface"]["depth_camera"]["pointcloud_mode"] == "euclidean_norm"
  assert payload["support_geometry_interface"]["depth_camera"]["timeout_ms"] == 500
  assert payload["support_geometry_interface"]["depth_camera"]["retain_last_valid_frame"] is True
  assert payload["support_geometry_interface"]["depth_camera"]["organized_pointcloud"] is True
