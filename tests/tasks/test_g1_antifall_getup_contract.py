from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity import mdp
from src.tasks.velocity.rl.antifall_runner import AntiFallOnPolicyRunner

TASK_ID = "Unitree-G1-AntiFall-GetUp"


def test_antifall_getup_task_is_registered_with_own_experiment() -> None:
  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is AntiFallOnPolicyRunner
  rl_cfg = load_rl_cfg(TASK_ID)
  assert rl_cfg.experiment_name == "g1_antifall_getup"
  assert rl_cfg.run_name == "antifall_getup"


def test_antifall_getup_env_combines_walking_push_and_fallen_recovery_contracts() -> None:
  cfg = load_env_cfg(TASK_ID)

  twist = cfg.commands["twist"]
  assert twist.rel_standing_envs <= 0.08
  assert twist.ranges.lin_vel_x[1] >= 1.5
  assert twist.ranges.lin_vel_y[0] < 0.0 < twist.ranges.lin_vel_y[1]
  assert "push_robot" in cfg.events
  assert cfg.events["push_robot"].func is mdp.push_by_setting_velocity_with_history

  reset_params = cfg.events["reset_base"].params
  assert 0.05 <= reset_params["hard_reset_prob"] <= 0.2
  assert reset_params["hard_pose_range"]["roll"][0] <= -2.0
  assert reset_params["hard_pose_range"]["pitch"][1] >= 2.0

  assert cfg.actions["joint_pos"].__class__.__name__ == "RecoveryHybridJointPositionActionCfg"
  assert cfg.actions["joint_pos"].max_delta <= 1.0
  assert cfg.observations["actor"].terms["actions"].func is mdp.host_effective_actions

  for reward_name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "recovery_quality",
    "recovery_completion_bonus",
    "host_lift_progress",
    "host_upright_progress",
    "host_hand_push",
    "getup_completion_bonus",
  ):
    assert reward_name in cfg.rewards
  for metric_name in (
    "controllable_locomotion",
    "disturbance_count",
    "recovery_success_count",
    "getup_success_count",
    "getup_latency",
  ):
    assert metric_name in cfg.metrics
  sensor_names = {sensor.name for sensor in cfg.scene.sensors}
  assert {"support_body_contact", "hand_ground_contact", "foot_geom_ground_contact"}.issubset(sensor_names)



def test_antifall_getup_stable_success_terms_do_not_share_mutable_scene_entity_cfgs() -> None:
  cfg = load_env_cfg(TASK_ID)
  terms = [
    cfg.rewards["host_task_reward"],
    cfg.rewards["getup_completion_bonus"],
    cfg.events["getup_assist_force"],
    cfg.metrics["getup_upright"],
    cfg.metrics["getup_success_count"],
    cfg.metrics["getup_latency"],
  ]

  foot_cfgs = [term.params["foot_asset_cfg"] for term in terms]
  torso_cfgs = [term.params["asset_cfg"] for term in terms]

  assert len({id(cfg) for cfg in foot_cfgs}) == len(foot_cfgs)
  assert len({id(cfg) for cfg in torso_cfgs}) == len(torso_cfgs)

  first_torso_cfg = torso_cfgs[0]
  first_torso_cfg.body_ids = [1]

  for other_torso_cfg in torso_cfgs[1:]:
    assert getattr(other_torso_cfg, "body_ids", None) != [1]


def test_antifall_getup_play_cfg_preserves_evaluation_disturbances_and_contact_headroom() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  assert "push_robot" in cfg.events
  assert cfg.sim.nconmax >= 256
  assert cfg.events["reset_base"].params["hard_reset_prob"] >= 0.05


def test_antifall_getup_play_randomizes_terrain_before_root_reset() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  reset_terms = [name for name, term in cfg.events.items() if term.mode == "reset"]

  assert "randomize_terrain" in reset_terms
  assert reset_terms.index("randomize_terrain") < reset_terms.index("reset_base")
  assert reset_terms.index("randomize_terrain") < reset_terms.index("reset_robot_joints")


def test_antifall_getup_uses_hybrid_action_to_preserve_warmstart_walking() -> None:
  from src.tasks.velocity.mdp.getup.actions import RecoveryHybridJointPositionActionCfg

  cfg = load_env_cfg(TASK_ID)
  action = cfg.actions["joint_pos"]

  assert isinstance(action, RecoveryHybridJointPositionActionCfg)
  assert action.use_default_offset is True
  assert action.recovery_use_default_offset is False
  assert action.recovery_window_s >= 2.0
  assert action.fallen_height_threshold <= 0.4
  assert action.fallen_tilt_threshold >= 0.7
  assert action.max_delta <= 1.0
  assert action.scale != 1.0


def test_antifall_getup_push_profile_marks_bfm_style_recovery_window() -> None:
  cfg = load_env_cfg(TASK_ID)
  push = cfg.events["push_robot"]

  assert push.params["recovery_window_s"] >= 2.0
  assert push.params["active"] is True

def test_antifall_getup_stall_guard_ignores_bfm_style_recovery_window() -> None:
  cfg = load_env_cfg(TASK_ID)
  stalled = cfg.terminations["stalled_getup"]

  assert stalled.params["recovery_grace_s"] >= 2.0
