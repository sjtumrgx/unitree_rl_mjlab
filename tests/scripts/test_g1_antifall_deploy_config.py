from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "deploy" / "robots" / "g1_antifall" / "config" / "config.yaml"
VELOCITY_DEPLOY_YAML_PATH = (
  ROOT / "deploy" / "robots" / "g1" / "config" / "policy" / "velocity" / "v0" / "params" / "deploy.yaml"
)
DEPLOY_YAML_PATH = (
  ROOT
  / "deploy"
  / "robots"
  / "g1_antifall"
  / "config"
  / "policy"
  / "antifall"
  / "stage4b"
  / "v0"
  / "params"
  / "deploy.yaml"
)


def test_g1_antifall_deploy_config_exposes_antifall_mode() -> None:
  payload = yaml.safe_load(CONFIG_PATH.read_text())
  fsm = payload["FSM"]
  assert "AntiFall" in fsm["_"]
  assert fsm["_"]["AntiFall"]["type"] == "RLBase"
  assert fsm["FixStand"]["keyboard_transitions"]["AntiFall"] == "v.on_pressed"
  assert fsm["FixStand"]["transitions"]["AntiFall"] == "RT + A.on_pressed"
  assert fsm["AntiFall"]["policy_dir"] == "config/policy/antifall/stage4b/v0"


def test_g1_antifall_repo_deploy_yaml_matches_antifall_contract() -> None:
  payload = yaml.safe_load(DEPLOY_YAML_PATH.read_text())
  velocity_payload = yaml.safe_load(VELOCITY_DEPLOY_YAML_PATH.read_text())
  observations = payload["observations"]
  assert tuple(observations) == (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
  )
  assert "gait_phase" not in observations
  assert observations["velocity_commands"]["params"] == {"command_name": "base_velocity"}
  assert {term_cfg["history_length"] for term_cfg in observations.values()} == {3}
  assert payload["joint_ids_map"] == velocity_payload["joint_ids_map"]
