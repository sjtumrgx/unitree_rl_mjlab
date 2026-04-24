from pathlib import Path

import yaml


DEPLOY_CONFIG = Path("deploy/robots/g1_getup/config/config.yaml")
DEPLOY_YAML = Path("deploy/robots/g1_getup/config/policy/getup/v0/params/deploy.yaml")


def test_g1_getup_deploy_config_uses_getup_runtime_entrypoint() -> None:
  cfg = yaml.safe_load(DEPLOY_CONFIG.read_text())
  fsm = cfg["FSM"]
  assert "GetUp" in fsm["_"]
  assert "Topology" + "GetUp" not in fsm["_"]
  assert fsm["FixStand"]["keyboard_transitions"]["GetUp"] == "g.on_pressed"
  assert fsm["GetUp"]["policy_dir"] == "config/policy/getup/v0"


def test_g1_getup_deploy_yaml_declares_getup_metadata() -> None:
  cfg = yaml.safe_load(DEPLOY_YAML.read_text())
  assert cfg["getup_terrain"] == "ground"
  assert cfg["host_source_task"] == "g1_ground"
  assert "getup_policy_interface" in cfg
  assert "support_geometry_interface" not in cfg
