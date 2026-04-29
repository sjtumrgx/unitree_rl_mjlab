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

    assert cfg.observations["actor"].history_length == 6
    assert getattr(cfg, "host_unactuated_timesteps") == 30

    action_cfg = cfg.actions["joint_pos"]
    assert isinstance(action_cfg, HostRelativeJointPositionActionCfg)
    assert action_cfg.unactuated_timesteps == 30
    assert action_cfg.scale == 1.0
    assert action_cfg.use_default_offset is False


def test_getup_assist_uses_host_curriculum_not_fixed_force_crutch() -> None:
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg

  cfg = unitree_g1_getup_env_cfg("ground")

  assert hasattr(mdp, "apply_host_getup_assist_force")
  assist = cfg.events["getup_assist_force"]
  assert assist.func is mdp.apply_host_getup_assist_force
  assert assist.params["initial_force_n"] == 100
  assert assist.params["force_decay_n"] == 20
  assert assist.params["action_scale_decay"] == 0.02
  assert assist.params["min_action_scale"] == 0.25
  assert assist.params["unactuated_timesteps"] == 30
  assert assist.params["orientation_projected_gravity_z_max"] == -0.8
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
    "host_feet_support": mdp.host_feet_support_reward,
    "host_target_standing": mdp.host_target_standing_reward,
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
