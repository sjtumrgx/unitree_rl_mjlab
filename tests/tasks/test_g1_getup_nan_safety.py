import torch


def test_getup_variants_enable_observation_sanitization_and_unstable_guard():
  from mjlab.managers.termination_manager import TerminationTermCfg
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import (
    GETUP_TERRAIN_VARIANTS,
    unitree_g1_getup_env_cfg,
  )

  for terrain in GETUP_TERRAIN_VARIANTS:
    cfg = unitree_g1_getup_env_cfg(terrain=terrain)

    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"

    unstable = cfg.terminations.get("unstable_state")
    assert isinstance(unstable, TerminationTermCfg)
    assert unstable.func is mdp.unstable_getup_state

    assert cfg.rewards["body_ang_vel"].func is mdp.bounded_body_angular_velocity_penalty
    assert cfg.rewards["angular_momentum"].func is mdp.bounded_angular_momentum_penalty
    assert cfg.rewards["joint_acc_l2"].func is mdp.bounded_joint_acc_l2
    assert cfg.rewards["action_rate_l2"].func is mdp.bounded_action_rate_after_lift


def test_getup_actor_and_critic_include_bfm_local_body_state_observation():
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg

  cfg = unitree_g1_getup_env_cfg("ground")

  for group_name in ("actor", "critic"):
    term = cfg.observations[group_name].terms.get("bfm_local_body_state")
    assert term is not None
    assert term.func is mdp.bfm_local_body_state

  actor_terms = cfg.observations["actor"].terms
  assert cfg.observations["actor"].history_length is None
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "getup_progress",
  ):
    assert actor_terms[term_name].history_length == 6
  assert actor_terms["bfm_local_body_state"].history_length == 0


def test_bfm_local_body_state_is_finite_and_heading_invariant():
  from types import SimpleNamespace

  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_mul
  from src.tasks.velocity.mdp.getup.observations import bfm_local_body_state

  yaw0 = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))[0]
  yaw90 = quat_from_euler_xyz(
    torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi / 2])
  )[0]
  local_body_pos = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.4, 0.0, 0.1],
      [0.0, -0.2, 0.3],
    ],
    dtype=torch.float32,
  )
  local_body_vel = torch.tensor(
    [
      [0.2, 0.0, 0.0],
      [0.1, 0.3, 0.0],
      [-0.2, 0.1, 0.05],
    ],
    dtype=torch.float32,
  )
  local_body_ang_vel = torch.tensor(
    [
      [0.0, 0.1, 0.2],
      [0.3, 0.0, 0.1],
      [0.2, -0.1, 0.0],
    ],
    dtype=torch.float32,
  )
  local_body_quat = yaw0.repeat(3, 1)
  root_pos = torch.tensor([[0.0, 0.0, 0.5], [2.0, -1.0, 1.5]], dtype=torch.float32)
  env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 1.0]], dtype=torch.float32)

  body_pos_0 = root_pos[0] + local_body_pos
  body_pos_1 = root_pos[1] + quat_apply(yaw90.repeat(3, 1), local_body_pos)
  body_vel_0 = local_body_vel
  body_vel_1 = quat_apply(yaw90.repeat(3, 1), local_body_vel)
  body_ang_vel_0 = local_body_ang_vel
  body_ang_vel_1 = quat_apply(yaw90.repeat(3, 1), local_body_ang_vel)
  body_quat_0 = local_body_quat
  body_quat_1 = quat_mul(yaw90.repeat(3, 1), local_body_quat)

  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=root_pos,
      root_link_quat_w=torch.stack([yaw0, yaw90], dim=0),
      body_link_pos_w=torch.stack([body_pos_0, body_pos_1], dim=0),
      body_link_quat_w=torch.stack([body_quat_0, body_quat_1], dim=0),
      body_link_lin_vel_w=torch.stack([body_vel_0, body_vel_1], dim=0),
      body_link_ang_vel_w=torch.stack([body_ang_vel_0, body_ang_vel_1], dim=0),
    )
  )
  env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot, "env_origins": env_origins})

  obs = bfm_local_body_state(env, asset_cfg=SceneEntityCfg("robot", body_ids=[0, 1, 2]))

  assert obs.shape == (2, 43)
  assert torch.isfinite(obs).all()
  assert obs[:, 0].tolist() == [0.5, 0.5]
  torch.testing.assert_close(obs[0], obs[1], atol=1e-5, rtol=1e-5)


def test_getup_ppo_uses_tight_action_and_entropy_bounds():
  from src.tasks.velocity.config.g1_getup.rl_cfg import unitree_g1_getup_ppo_runner_cfg

  cfg = unitree_g1_getup_ppo_runner_cfg("ground")

  assert cfg.clip_actions is not None
  assert cfg.clip_actions <= 5.0
  assert cfg.actor.distribution_cfg["init_std"] <= 0.6
  assert cfg.algorithm.entropy_coef <= 0.002


def test_policy_action_sanitizer_clamps_nan_and_inf():
  from src.tasks.velocity.rl.safety import sanitize_policy_actions

  actions = torch.tensor([[float("nan"), float("inf"), -float("inf"), 3.0, -6.0]])
  sanitized = sanitize_policy_actions(actions, clip_actions=5.0)

  assert torch.isfinite(sanitized).all()
  assert sanitized.tolist() == [[0.0, 5.0, -5.0, 3.0, -5.0]]

# HoST parity regression tests added for G1 GetUp reward/curriculum repair.
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


_REPO_ROOT = Path(__file__).resolve().parents[2]
_G1_23DOF_XML = _REPO_ROOT / "src/assets/robots/unitree_g1/xmls/g1_23dof.xml"


def _active_g1_23dof_joint_names() -> set[str]:
  root = ET.parse(_G1_23DOF_XML).getroot()
  return {
    joint.attrib["name"]
    for joint in root.findall(".//joint")
    if "name" in joint.attrib and joint.attrib["name"] != "floating_base_joint"
  }


def _joint_names_from_reward_params(cfg) -> set[str]:
  names: set[str] = set()
  for term in cfg.rewards.values():
    if not isinstance(term, RewardTermCfg):
      continue
    params = getattr(term, "params", {}) or {}
    joint_names = params.get("joint_names")
    if joint_names is not None:
      names.update(joint_names)
    joint_targets = params.get("target_joint_angles")
    if isinstance(joint_targets, dict):
      names.update(joint_targets)
  return names


def test_getup_actor_history_and_unactuated_contract_match_host_ground() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import GETUP_TERRAIN_VARIANTS, unitree_g1_getup_env_cfg
  from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg

  for terrain in GETUP_TERRAIN_VARIANTS:
    cfg = unitree_g1_getup_env_cfg(terrain=terrain)

    assert cfg.observations["actor"].history_length is None
    assert getattr(cfg, "host_unactuated_timesteps") == 30

    action_cfg = cfg.actions["joint_pos"]
    assert isinstance(action_cfg, HostRelativeJointPositionActionCfg)
    assert action_cfg.unactuated_timesteps == 30
    assert action_cfg.scale == 1.0
    assert action_cfg.use_default_offset is False
    assert action_cfg.max_delta == 1.0


def test_getup_stall_guard_allows_multi_second_recovery_attempts() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_amp_env_cfg, unitree_g1_getup_env_cfg
  from src.tasks.velocity.mdp.getup import stalled_getup_progress

  # Local fall->stand AMP clips include valid recoveries that take up to about
  # 11.2s after resampling to the 50 Hz env step.  The stall guard must leave
  # enough room for those trajectories and for early PPO exploration before it
  # declares "no progress".
  min_demo_recovery_s = 12.0
  cfgs = (
    unitree_g1_getup_env_cfg("ground"),
    unitree_g1_getup_amp_env_cfg(demo_data_dir="/tmp/g1_getup_amp_fixture"),
  )

  for cfg in cfgs:
    stalled = cfg.terminations["stalled_getup"]
    assert stalled.func is stalled_getup_progress
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert cfg.episode_length_s >= min_demo_recovery_s
    assert stalled.params["min_steps_before_check"] >= int(min_demo_recovery_s / step_dt)
    assert stalled.params["target_height"] >= 0.55


def test_getup_action_observations_use_executed_effective_actions_for_normal_and_amp() -> None:
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import (
    unitree_g1_getup_amp_env_cfg,
    unitree_g1_getup_env_cfg,
  )

  effective_actions = getattr(mdp, "host_effective_actions", None)
  assert effective_actions is not None

  cfgs = (
    unitree_g1_getup_env_cfg("ground"),
    unitree_g1_getup_amp_env_cfg(demo_data_dir="/tmp/g1_getup_amp_fixture"),
  )
  for cfg in cfgs:
    assert cfg.observations["actor"].terms["actions"].func is effective_actions
    assert cfg.observations["critic"].terms["actions"].func is effective_actions


def test_getup_assist_uses_host_curriculum_not_fixed_force_crutch() -> None:
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg

  cfg = unitree_g1_getup_env_cfg("ground")

  assert hasattr(mdp, "apply_host_getup_assist_force")
  assist = cfg.events["getup_assist_force"]
  assert assist.func is mdp.apply_host_getup_assist_force
  assert assist.params["initial_force_n"] == 120
  assert assist.params["force_decay_n"] == 20
  assert assist.params["action_scale_decay"] == 0.02
  assert assist.params["min_action_scale"] == 0.25
  assert assist.params["unactuated_timesteps"] == 30
  assert 0.55 <= assist.params["success_height_threshold"] < 0.9
  assert assist.params["no_orientation_gate"] is True
  assert cfg.metrics["getup_assist_force_n"].func is mdp.getup_assist_force_n
  assert cfg.metrics["getup_action_rescale"].func is mdp.getup_action_rescale

  play_cfg = unitree_g1_getup_env_cfg("ground", play=True)
  assert "getup_assist_force" not in play_cfg.events


def test_getup_reward_stack_replaces_reward_hacks_with_host_composite_terms() -> None:
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg

  cfg = unitree_g1_getup_env_cfg("ground")
  reward_names = set(cfg.rewards)

  # These terms paid the failed policy for looking upright/raising the torso without
  # requiring foot support or actual standing progress.
  disallowed = {
    "getup_posture_reward",
    "getup_torso_lift_reward",
    "getup_facing_up_reward",
    "getup_orientation_phase_bonus",
    "getup_height_progress_reward",
    "getup_phase_bonus",
    "getup_demo_pose_reward",
    "upright_recoverability",
    "recovery_quality",
    "standing_stability",
    "recovery_completion_bonus",
    "foot_gait",
    "foot_clearance",
    "foot_slip",
    "soft_landing",
    "pose",
    "body_orientation_l2",
  }
  assert reward_names.isdisjoint(disallowed)

  expected_funcs = {
    "host_task_reward": mdp.host_getup_task_reward,
    "host_action_smoothness": mdp.host_action_smoothness_penalty,
    "host_joint_tracking": mdp.host_joint_tracking_penalty,
    "host_style_pose": mdp.host_style_pose_reward,
    "host_upright_progress": mdp.host_upright_progress_reward,
    "host_support_relief": mdp.host_support_relief_reward,
    "host_feet_support": mdp.host_feet_support_reward,
    "host_target_standing": mdp.host_target_standing_reward,
    "getup_completion_bonus": mdp.getup_completion_bonus,
  }
  for name, func in expected_funcs.items():
    assert name in cfg.rewards
    assert cfg.rewards[name].func is func


def test_getup_configured_joint_names_exist_in_active_23dof_model() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import GETUP_TERRAIN_VARIANTS, unitree_g1_getup_env_cfg

  active_joint_names = _active_g1_23dof_joint_names()
  assert "waist_pitch_joint" not in active_joint_names  # guards against using scene/full model by mistake

  for terrain in GETUP_TERRAIN_VARIANTS:
    cfg = unitree_g1_getup_env_cfg(terrain=terrain)
    referenced = _joint_names_from_reward_params(cfg)
    missing = referenced - active_joint_names
    assert not missing, f"{terrain} references non-active G1 23DoF joints: {sorted(missing)}"


class _FakeRobot:
  def __init__(self, num_envs: int = 2):
    self.joint_names = ["j0", "j1", "j2"]
    self.data = SimpleNamespace(
      joint_pos=torch.zeros(num_envs, 3),
      encoder_bias=torch.zeros(num_envs, 3),
    )
    self.targets: torch.Tensor | None = None

  def find_joints_by_actuator_names(self, actuator_names):
    del actuator_names
    return [0, 1, 2], self.joint_names

  def set_joint_position_target(self, target, joint_ids):
    del joint_ids
    self.targets = target.clone()


class _FakeActionEnv:
  def __init__(self):
    self.num_envs = 2
    self.device = "cpu"
    self.episode_length_buf = torch.tensor([0, 31], dtype=torch.long)
    self.scene = {"robot": _FakeRobot(self.num_envs)}


def test_host_relative_action_zeros_delta_during_unactuated_startup() -> None:
  from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg

  env = _FakeActionEnv()
  action = HostRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    unactuated_timesteps=30,
  ).build(env)

  raw = torch.tensor([[2.0, -2.0, 0.5], [0.25, -0.25, 0.75]])
  action.process_actions(raw)
  action.apply_actions()

  target = env.scene["robot"].targets
  assert target is not None
  assert torch.allclose(action.raw_action[0], raw[0])
  assert torch.allclose(target[0], torch.zeros(3))
  assert torch.allclose(target[1], raw[1])


def test_host_relative_action_records_effective_history_after_warmup_masking() -> None:
  from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg

  env = _FakeActionEnv()
  action = HostRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    unactuated_timesteps=30,
  ).build(env)

  first_raw = torch.tensor([[2.0, -2.0, 0.5], [0.25, -0.25, 0.75]])
  second_raw = torch.tensor([[-3.0, 1.5, 2.0], [1.0, 0.5, -0.5]])
  action.process_actions(first_raw)
  action.process_actions(second_raw)

  effective_action = getattr(action, "effective_action", None)
  prev_effective_action = getattr(action, "prev_effective_action", None)
  prev_prev_effective_action = getattr(action, "prev_prev_effective_action", None)
  assert effective_action is not None
  assert prev_effective_action is not None
  assert prev_prev_effective_action is not None

  assert torch.allclose(action.raw_action[0], second_raw[0])
  assert torch.allclose(effective_action[0], torch.zeros(3))
  assert torch.allclose(prev_effective_action[0], torch.zeros(3))
  assert torch.allclose(prev_prev_effective_action[0], torch.zeros(3))
  assert torch.allclose(effective_action[1], second_raw[1])
  assert torch.allclose(prev_effective_action[1], first_raw[1])
  assert torch.allclose(prev_prev_effective_action[1], torch.zeros(3))
  assert torch.allclose(env._host_getup_effective_action, effective_action)
  assert torch.allclose(env._host_getup_prev_effective_action, prev_effective_action)
  assert torch.allclose(env._host_getup_prev_prev_effective_action, prev_prev_effective_action)


def test_host_relative_action_clamps_current_pose_delta_after_startup() -> None:
  from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg

  env = _FakeActionEnv()
  env.episode_length_buf[:] = 31
  action = HostRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    unactuated_timesteps=30,
    max_delta=0.75,
  ).build(env)

  raw = torch.tensor([[2.0, -2.0, 0.5], [0.25, -0.25, 1.5]])
  action.process_actions(raw)
  action.apply_actions()

  target = env.scene["robot"].targets
  assert target is not None
  assert torch.allclose(
    target,
    torch.tensor([[0.75, -0.75, 0.5], [0.25, -0.25, 0.75]]),
  )
  assert torch.allclose(env._host_getup_joint_position_delta, target)


def test_no_assist_episode_uses_play_action_scale_instead_of_curriculum_rescale() -> None:
  from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg

  env = _FakeActionEnv()
  env.episode_length_buf[:] = 31
  env._host_getup_curriculum_state = {
    "action_rescale": torch.tensor([0.25, 0.25]),
    "episode_force_scale": torch.tensor([0.0, 1.0]),
  }
  action = HostRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    unactuated_timesteps=30,
  ).build(env)

  raw = torch.tensor([[0.8, -0.4, 0.2], [0.8, -0.4, 0.2]])
  action.process_actions(raw)
  action.apply_actions()

  assert torch.allclose(
    env._host_getup_joint_position_delta,
    torch.tensor([[0.8, -0.4, 0.2], [0.2, -0.1, 0.05]]),
  )


class _FakeContactSensor:
  def __init__(self, found: torch.Tensor):
    self.data = SimpleNamespace(found=found)


class _FakeRewardRobot:
  def __init__(self, torso_height: torch.Tensor, projected_gravity_b: torch.Tensor):
    self.joint_names = [
      "left_hip_yaw_joint",
      "left_hip_roll_joint",
      "left_hip_pitch_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_yaw_joint",
      "right_hip_roll_joint",
      "right_hip_pitch_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
    ]
    n = torso_height.numel()
    self.data = SimpleNamespace(
      body_link_pos_w=torso_height.view(n, 1, 1).repeat(1, 1, 3),
      projected_gravity_b=projected_gravity_b,
      joint_pos=torch.zeros(n, len(self.joint_names)),
      default_joint_pos=torch.zeros(n, len(self.joint_names)),
    )
    self.data.body_link_pos_w[:, :, 2] = torso_height.view(n, 1)


class _FakeRewardEnv:
  def __init__(self):
    self.num_envs = 2
    self.device = "cpu"
    self.scene = {
      "robot": _FakeRewardRobot(
        torso_height=torch.tensor([0.22, 0.72]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -0.99], [0.0, 0.0, -1.0]]),
      ),
      "feet_ground_contact": _FakeContactSensor(
        torch.tensor([[[0.0], [0.0]], [[1.0], [1.0]]])
      ),
      "support_body_contact": _FakeContactSensor(
        torch.tensor([[[0.0]], [[0.0]]])
      ),
      "env_origins": torch.zeros(2, 3),
    }


def test_host_task_reward_does_not_pay_upright_torso_without_feet_supported_height() -> None:
  from src.tasks.velocity import mdp

  env = _FakeRewardEnv()
  reward = mdp.host_getup_task_reward(
    env,
    feet_sensor_name="feet_ground_contact",
    body_sensor_name="support_body_contact",
    asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
  )

  assert reward[0].item() < 0.05
  assert reward[1].item() > 0.5


def test_host_lift_progress_does_not_pay_tall_non_upright_postures_like_recovery() -> None:
  from src.tasks.velocity import mdp

  env = _FakeRewardEnv()
  env.scene["robot"] = _FakeRewardRobot(
    torso_height=torch.tensor([0.50, 0.50]),
    projected_gravity_b=torch.tensor([[0.95, 0.0, -0.10], [0.0, 0.0, -0.98]]),
  )

  reward = mdp.host_getup_lift_progress_reward(
    env,
    orientation_floor=0.0,
    asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
  )

  assert reward[0].item() < 0.35
  assert reward[1].item() > 0.75


def test_host_upright_progress_rewards_rotation_when_torso_is_lifted() -> None:
  from src.tasks.velocity import mdp

  env = _FakeRewardEnv()
  env.scene["robot"] = _FakeRewardRobot(
    torso_height=torch.tensor([0.50, 0.50]),
    projected_gravity_b=torch.tensor([[0.95, 0.0, -0.10], [0.0, 0.0, -0.98]]),
  )

  reward = mdp.host_upright_progress_reward(
    env,
    alignment_floor=0.0,
    asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
  )

  assert reward[0].item() < 0.2
  assert reward[1].item() > reward[0].item() * 5.0


def test_host_support_relief_rewards_unloading_body_contacts_with_foot_support() -> None:
  from src.tasks.velocity import mdp

  env = _FakeRewardEnv()
  env.scene["robot"] = _FakeRewardRobot(
    torso_height=torch.tensor([0.56, 0.56]),
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
  )
  env.scene["feet_ground_contact"] = _FakeContactSensor(
    torch.tensor([[[1.0], [1.0]], [[1.0], [1.0]]])
  )
  env.scene["support_body_contact"] = _FakeContactSensor(
    torch.tensor(
      [
        [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
        [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
      ]
    )
  )

  reward = mdp.host_support_relief_reward(
    env,
    feet_sensor_name="feet_ground_contact",
    body_sensor_name="support_body_contact",
    max_body_support_count=8.0,
    alignment_floor=0.0,
    asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
  )

  assert reward[0].item() == 0.0
  assert reward[1].item() > 0.5


def test_host_target_standing_rewards_reachable_success_band() -> None:
  from src.tasks.velocity import mdp

  env = _FakeRewardEnv()
  env.scene["robot"] = _FakeRewardRobot(
    torso_height=torch.tensor([0.50, 0.56]),
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
  )

  reward = mdp.host_target_standing_reward(
    env,
    feet_sensor_name="feet_ground_contact",
    body_sensor_name="support_body_contact",
    base_height_target=0.75,
    target_base_height_phase3=0.65,
    standing_gate_start_height=0.45,
    max_body_support_count=8.0,
    asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",)),
  )

  assert reward[0].item() >= 0.0
  assert reward[1].item() > reward[0].item()


def test_getup_reward_stack_requires_upright_progress_for_dense_lift_reward() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg

  cfg = unitree_g1_getup_env_cfg("ground")
  lift_params = cfg.rewards["host_lift_progress"].params

  assert lift_params["orientation_floor"] >= 0.0
  assert cfg.rewards["host_upright_progress"].params["alignment_floor"] >= 0.0
  assert cfg.rewards["host_support_relief"].params["alignment_floor"] >= 0.0


def test_getup_reward_stack_targets_no_assist_transfer_before_full_stand() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import GETUP_SUCCESS_TORSO_HEIGHT, unitree_g1_getup_env_cfg

  cfg = unitree_g1_getup_env_cfg("ground")
  target = cfg.rewards["host_target_standing"]
  upright = cfg.rewards["host_upright_progress"]
  support_relief = cfg.rewards["host_support_relief"]
  assist_params = cfg.events["getup_assist_force"].params

  assert target.weight >= 2.0
  assert target.params["standing_gate_start_height"] <= GETUP_SUCCESS_TORSO_HEIGHT
  assert target.params["max_body_support_count"] >= 8.0
  assert upright.weight >= 1.0
  assert support_relief.weight >= 1.0
  assert assist_params["no_assist_probability"] >= 0.25


def test_host_action_smoothness_penalty_ignores_raw_policy_actions_not_executed_in_warmup() -> None:
  from src.tasks.velocity import mdp

  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    action_manager=SimpleNamespace(
      action=torch.tensor([[8.0, -8.0, 4.0]]),
      prev_action=torch.tensor([[-8.0, 8.0, -4.0]]),
      prev_prev_action=torch.tensor([[8.0, -8.0, 4.0]]),
    ),
    _host_getup_effective_action=torch.zeros(1, 3),
    _host_getup_prev_effective_action=torch.zeros(1, 3),
    _host_getup_prev_prev_effective_action=torch.zeros(1, 3),
  )

  penalty = mdp.host_action_smoothness_penalty(env)

  assert torch.allclose(penalty, torch.zeros(1))
