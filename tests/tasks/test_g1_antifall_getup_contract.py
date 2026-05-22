from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity import mdp
from src.tasks.velocity.rl.antifall_runner import AntiFallOnPolicyRunner

TASK_ID = "Unitree-G1-AntiFall-GetUp"
WARMUP_TASK_ID = "Unitree-G1-AntiFall-GetUp-RecoveryWarmup"


def test_antifall_getup_task_is_registered_with_own_experiment() -> None:
  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is AntiFallOnPolicyRunner
  rl_cfg = load_rl_cfg(TASK_ID)
  assert rl_cfg.experiment_name == "g1_antifall_getup"
  assert rl_cfg.run_name == "antifall_getup"


def test_antifall_getup_recovery_warmup_task_is_registered() -> None:
  assert WARMUP_TASK_ID in list_tasks()
  assert load_runner_cls(WARMUP_TASK_ID) is AntiFallOnPolicyRunner
  rl_cfg = load_rl_cfg(WARMUP_TASK_ID)
  assert rl_cfg.experiment_name == "g1_antifall_getup"
  assert rl_cfg.run_name == "recovery_warmup"
  assert rl_cfg.algorithm.learning_rate <= 1.0e-4


def test_antifall_getup_recovery_warmup_keeps_final_actor_contract() -> None:
  warmup = load_env_cfg(WARMUP_TASK_ID)
  final = load_env_cfg(TASK_ID)

  assert tuple(warmup.observations["actor"].terms) == tuple(final.observations["actor"].terms)
  assert warmup.observations["actor"].history_length == final.observations["actor"].history_length
  assert warmup.actions["joint_pos"].__class__ is final.actions["joint_pos"].__class__
  assert warmup.actions["joint_pos"].max_delta == final.actions["joint_pos"].max_delta


def test_antifall_getup_recovery_warmup_is_fallen_recovery_only() -> None:
  cfg = load_env_cfg(WARMUP_TASK_ID)

  twist = cfg.commands["twist"]
  assert twist.rel_standing_envs == 1.0
  assert twist.ranges.lin_vel_x == (0.0, 0.0)
  assert twist.ranges.lin_vel_y == (0.0, 0.0)
  assert twist.ranges.ang_vel_z == (0.0, 0.0)
  assert "push_robot" not in cfg.events
  assert "mid_episode_forced_fall" not in cfg.events

  reset_base = cfg.events["reset_base"]
  assert reset_base.func is mdp.reset_root_state_from_presets
  assert reset_base.params["preset_weight_stages"][0]["weights"][-1] > 0.0
  assert cfg.events["reset_robot_joints"].func is mdp.reset_joints_from_presets
  assert cfg.terminations["stalled_getup"].params["recovery_grace_s"] == 0.0

  assert cfg.rewards["host_lift_progress"].func is not mdp.recovery_phase_reward
  assert "track_linear_velocity" not in cfg.rewards
  assert "post_recovery_resume_locomotion" not in cfg.rewards



def test_antifall_getup_runner_uses_conservative_warmstart_finetune_hyperparams() -> None:
  rl_cfg = load_rl_cfg(TASK_ID)

  assert rl_cfg.algorithm.learning_rate <= 1.0e-4
  assert rl_cfg.algorithm.desired_kl <= 0.003
  assert rl_cfg.actor.distribution_cfg["init_std"] <= 0.5

def test_antifall_getup_env_combines_walking_push_and_fallen_recovery_contracts() -> None:
  cfg = load_env_cfg(TASK_ID)

  twist = cfg.commands["twist"]
  assert twist.rel_standing_envs <= 0.08
  assert twist.ranges.lin_vel_x[1] >= 1.5
  assert twist.ranges.lin_vel_y[0] < 0.0 < twist.ranges.lin_vel_y[1]
  assert "push_robot" in cfg.events
  assert cfg.events["push_robot"].func is mdp.push_by_setting_velocity_with_history

  reset_params = cfg.events["reset_base"].params
  assert reset_params["hard_reset_prob"] == 0.0
  schedule = reset_params["hard_reset_prob_schedule"]
  assert schedule[0] == {"step": 0, "prob": 0.02}
  assert 0.05 <= schedule[-1]["prob"] <= 0.2
  assert reset_params["presets"][1]["pose_range"]["z"] == (-0.55, -0.45)
  assert reset_params["preset_weight_stages"][0]["weights"][-1] > 0.0

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




def test_antifall_getup_training_hard_resets_ramp_after_walking_warm_start() -> None:
  cfg = load_env_cfg(TASK_ID)
  reset_params = cfg.events["reset_base"].params

  assert cfg.events["reset_base"].func is mdp.reset_root_state_mixed_from_presets
  assert reset_params["hard_reset_prob"] == 0.0
  assert reset_params["hard_reset_prob_schedule"][0] == {"step": 0, "prob": 0.02}
  assert reset_params["hard_reset_prob_schedule"][1]["step"] == 300
  assert reset_params["hard_reset_prob_schedule"][-1]["prob"] <= 0.15


def test_antifall_getup_hard_resets_pair_fallen_root_with_getup_joint_presets() -> None:
  cfg = load_env_cfg(TASK_ID)

  reset_base = cfg.events["reset_base"]
  reset_joints = cfg.events["reset_robot_joints"]

  assert reset_base.func is mdp.reset_root_state_mixed_from_presets
  assert "hard_pose_range" not in reset_base.params
  assert reset_base.params["preset_weight_stages"][0]["weights"][-1] > 0.0
  assert reset_joints.func is mdp.reset_joints_mixed_by_antifall_state
  assert reset_joints.params["nominal_position_range"] == (-0.0, 0.0)
  assert reset_joints.params["nominal_velocity_range"] == (-0.0, 0.0)
  assert reset_joints.params["preset_position_noise_range"] == (-0.05, 0.05)




def test_antifall_getup_uses_getup_safe_recovery_regularizers() -> None:
  cfg = load_env_cfg(TASK_ID)

  assert cfg.rewards["body_ang_vel"].func is mdp.bounded_body_angular_velocity_penalty
  assert cfg.rewards["angular_momentum"].func is mdp.bounded_angular_momentum_penalty
  assert cfg.rewards["joint_acc_l2"].func is mdp.bounded_joint_acc_l2
  assert cfg.rewards["action_rate_l2"].func is mdp.bounded_action_rate_after_lift
  assert cfg.rewards["joint_pos_limits"].func is mdp.joint_pos_limits_after_support
  assert cfg.rewards["joint_pos_limits"].params["body_sensor_name"] == "support_body_contact"
  assert cfg.rewards["self_collisions"].func is mdp.self_collision_cost_after_support
  assert "support_body_contact_penalty_after_lift" in cfg.rewards
  assert "pelvis_clearance_penalty" in cfg.rewards

def test_antifall_getup_training_has_mid_episode_paired_forced_fall_exposure() -> None:
  cfg = load_env_cfg(TASK_ID)

  event = cfg.events["mid_episode_forced_fall"]

  assert event.mode == "interval"
  assert event.func is mdp.reset_paired_fallen_state_from_presets
  assert event.interval_range_s[0] >= 5.0
  assert event.params["presets"][1]["pose_range"]["z"] == (-0.55, -0.45)
  assert event.params["preset_weight_stages"][0]["weights"][-1] > 0.0
  assert event.params["joint_position_noise_range"] == (-0.05, 0.05)
  assert event.params["reset_actions"] is True


def test_antifall_getup_play_does_not_add_training_forced_fall_curriculum() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)

  assert "mid_episode_forced_fall" not in cfg.events


def test_paired_fallen_state_reset_synchronizes_root_joints_and_action_history(monkeypatch) -> None:
  calls = []

  def fake_reset_root(env, env_ids, **kwargs):
    calls.append(("root", env_ids.tolist(), kwargs))

  def fake_reset_joints(env, env_ids, **kwargs):
    calls.append(("joints", env_ids.tolist(), kwargs))

  monkeypatch.setattr(mdp, "reset_root_state_from_presets", fake_reset_root)
  monkeypatch.setattr(mdp, "reset_joints_from_presets", fake_reset_joints)

  import src.tasks.velocity.mdp.getup.events as getup_events
  import torch
  from types import SimpleNamespace

  monkeypatch.setattr(getup_events, "reset_root_state_from_presets", fake_reset_root)
  monkeypatch.setattr(getup_events, "reset_joints_from_presets", fake_reset_joints)
  action_resets = []
  env_ids = torch.tensor([0, 2], dtype=torch.long)
  env = SimpleNamespace(action_manager=SimpleNamespace(reset=lambda env_ids=None: action_resets.append(env_ids.clone())))

  mdp.reset_paired_fallen_state_from_presets(
    env,
    env_ids,
    presets=("preset",),
    preset_weight_stages=({"step": 0, "weights": (1.0,)},),
    velocity_range={"x": (0.0, 0.0)},
    joint_position_noise_range=(0.0, 0.0),
    joint_velocity_range=(0.0, 0.0),
  )

  assert [name for name, _, _ in calls] == ["root", "joints"]
  assert calls[0][1] == [0, 2]
  assert calls[0][2]["presets"] == ("preset",)
  assert calls[1][2]["position_noise_range"] == (0.0, 0.0)
  assert len(action_resets) == 1
  assert action_resets[0].tolist() == [0, 2]


def test_scheduled_hard_reset_probability_uses_latest_elapsed_step() -> None:
  from src.tasks.velocity.mdp.anti_fall.events import scheduled_hard_reset_prob

  schedule = (
    {"step": 0, "prob": 0.0},
    {"step": 10, "prob": 0.05},
    {"step": 20, "prob": 0.15},
  )

  assert scheduled_hard_reset_prob(0.3, schedule, common_step_counter=0) == 0.0
  assert scheduled_hard_reset_prob(0.3, schedule, common_step_counter=9) == 0.0
  assert scheduled_hard_reset_prob(0.3, schedule, common_step_counter=10) == 0.05
  assert scheduled_hard_reset_prob(0.3, schedule, common_step_counter=25) == 0.15
  assert scheduled_hard_reset_prob(0.3, None, common_step_counter=25) == 0.3


def test_antifall_getup_gates_host_getup_rewards_to_recovery_phase() -> None:
  cfg = load_env_cfg(TASK_ID)

  for reward_name in (
    "host_task_reward",
    "host_lift_progress",
    "host_upright_progress",
    "host_support_relief",
    "host_action_smoothness",
    "host_joint_tracking",
    "host_style_pose",
    "host_feet_support",
    "getup_completion_bonus",
    "host_hand_contact_after_stand",
    "host_foot_orientation_penalty",
    "host_ankle_deviation_penalty",
  ):
    term = cfg.rewards[reward_name]
    assert term.func is mdp.recovery_phase_reward
    assert "reward_func" in term.params
    assert term.params["fallen_height_threshold"] <= 0.4
    assert term.params["fallen_tilt_threshold"] >= 0.7
    assert term.params["include_disturbance_window"] is False
    assert term.params["window_s"] == 0.0



def test_antifall_getup_trains_post_recovery_resume_locomotion() -> None:
  cfg = load_env_cfg(TASK_ID)

  resume = cfg.rewards["post_recovery_resume_locomotion"]
  assert resume.func is mdp.post_recovery_resume_locomotion
  assert resume.weight >= 1.0
  assert resume.params["resume_window_s"] > resume.params["recovery_window_s"]
  assert resume.params["fallen_height_threshold"] <= 0.4
  assert resume.params["fallen_tilt_threshold"] >= 0.7

  assert "post_recovery_resume_locomotion" in cfg.metrics

def test_antifall_getup_recovery_rewards_ignore_plain_push_windows() -> None:
  cfg = load_env_cfg(TASK_ID)

  recovery_reward = cfg.rewards["recovery_quality"]
  assert recovery_reward.func is mdp.recovery_quality
  assert recovery_reward.params["require_fallen_or_near_failure"] is True
  assert recovery_reward.params["fallen_height_threshold"] <= 0.4
  assert recovery_reward.params["fallen_tilt_threshold"] >= 0.7

  completion = cfg.rewards["recovery_completion_bonus"]
  assert completion.func is mdp.recovery_completion_bonus
  assert completion.params["require_fallen_or_near_failure"] is True


def test_antifall_getup_assist_is_recovery_phase_gated() -> None:
  cfg = load_env_cfg(TASK_ID)
  assist = cfg.events["getup_assist_force"]

  assert assist.params["recovery_phase_only"] is True
  assert assist.params["fallen_height_threshold"] <= 0.4
  assert assist.params["fallen_tilt_threshold"] >= 0.7
  assert assist.params["include_disturbance_window"] is False

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


def test_antifall_getup_can_disable_hard_resets_for_walk_then_recover_gate() -> None:
  from src.tasks.velocity.config.g1_antifall.env_cfgs import unitree_g1_antifall_getup_env_cfg

  cfg = unitree_g1_antifall_getup_env_cfg(play=True, hard_reset_prob=0.0)

  assert cfg.events["reset_base"].params["hard_reset_prob"] == 0.0
  assert "push_robot" in cfg.events


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


def test_antifall_getup_training_push_profile_is_warmstart_friendly() -> None:
  cfg = load_env_cfg(TASK_ID)
  push = cfg.events["push_robot"]
  profile = push.params["velocity_range"]

  assert push.params["recovery_window_s"] >= 2.0
  assert push.params["active"] is True
  assert push.interval_range_s[0] >= 5.0
  assert profile["x"] == (-0.75, 0.75)
  assert profile["y"] == (-0.75, 0.75)
  assert profile["z"] == (-0.5, 0.5)
  assert profile["yaw"] == (-1.0, 1.0)


def test_antifall_getup_play_keeps_strong_push_gate_coverage() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  push = cfg.events["push_robot"]
  profile = push.params["velocity_range"]

  assert push.interval_range_s[1] <= 3.5
  assert profile["x"] == (-1.4, 1.4)
  assert profile["y"] == (-1.4, 1.4)
  assert profile["yaw"] == (-1.4, 1.4)

def test_antifall_getup_stall_guard_ignores_bfm_style_recovery_window() -> None:
  cfg = load_env_cfg(TASK_ID)
  stalled = cfg.terminations["stalled_getup"]

  assert stalled.params["recovery_grace_s"] >= 2.0
