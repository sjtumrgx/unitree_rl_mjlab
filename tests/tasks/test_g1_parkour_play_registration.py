from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.parkour.contract import (
  ACTION_SIZE,
  PARKOUR_SCENE_SENSOR_REMAP,
  TRAINING_JOINT_NAMES,
  assert_no_stale_sensor_references,
  load_deploy_contract,
)
from src.tasks.velocity.config.g1_parkour.env_cfgs import PARKOUR_FLAT_DEBUG_TASK_ID
from src.tasks.velocity.rl import VelocityOnPolicyRunner


def test_g1_parkour_flat_debug_task_is_registered() -> None:
  assert PARKOUR_FLAT_DEBUG_TASK_ID in list_tasks()


def test_g1_parkour_flat_debug_cfg_uses_parkour_robot_and_no_stale_scene_sensors() -> None:
  cfg = load_env_cfg(PARKOUR_FLAT_DEBUG_TASK_ID, play=True)
  assert "robot" in cfg.scene.entities
  assert getattr(cfg, "g1_parkour_flat_debug") is True
  assert_no_stale_sensor_references(cfg)

  actor_terms = cfg.observations["actor"].terms
  assert actor_terms["base_ang_vel"].params["sensor_name"] == PARKOUR_SCENE_SENSOR_REMAP["robot/imu_ang_vel"]
  assert set(actor_terms) == {
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
  }


def test_g1_parkour_action_contract_matches_deploy_yaml() -> None:
  cfg = load_env_cfg(PARKOUR_FLAT_DEBUG_TASK_ID, play=True)
  contract = load_deploy_contract()
  action = cfg.actions["joint_pos"]

  assert isinstance(action, JointPositionActionCfg)
  assert len(action.scale) == ACTION_SIZE
  for name, scale in zip(TRAINING_JOINT_NAMES, contract.action_scales, strict=True):
    assert action.scale[name] == scale


def test_g1_parkour_rl_cfg_loads_as_dedicated_onnx_debug_task() -> None:
  rl_cfg = load_rl_cfg(PARKOUR_FLAT_DEBUG_TASK_ID)
  assert rl_cfg.experiment_name == "g1_parkour_flat_debug"
  assert load_runner_cls(PARKOUR_FLAT_DEBUG_TASK_ID) is VelocityOnPolicyRunner
