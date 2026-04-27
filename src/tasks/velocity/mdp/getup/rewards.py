from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import torch
import numpy as np

from mjlab.envs import mdp as envs_mdp
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp as base_envs_mdp

from ..rewards import (
  angular_momentum_penalty as _base_angular_momentum_penalty,
  body_angular_velocity_penalty as _base_body_angular_velocity_penalty,
  stand_still as _base_stand_still,
  track_angular_velocity as _base_track_angular_velocity,
  track_linear_velocity as _base_track_linear_velocity,
  self_collision_cost as _base_self_collision_cost,
)
from .metrics import _upright_alignment, getup_upright, pelvis_clearance_violation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_TORSO_ASSET_CFG = SceneEntityCfg("robot", body_names=("torso_link",))
_JOINT_ASSET_CFG = SceneEntityCfg("robot")


def _bounded_nonnegative_penalty(
  value: torch.Tensor,
  *,
  max_penalty: float,
) -> torch.Tensor:
  """Clamp positive penalty signals before they dominate PPO returns."""

  max_value = float(max_penalty)
  finite = torch.nan_to_num(value, nan=max_value, posinf=max_value, neginf=max_value)
  return torch.clamp(finite, min=0.0, max=max_value)


def bounded_body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  max_penalty: float = 400.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  penalty = _base_body_angular_velocity_penalty(env, asset_cfg=asset_cfg)
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def bounded_angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_penalty: float = 1000.0,
) -> torch.Tensor:
  penalty = _base_angular_momentum_penalty(env, sensor_name=sensor_name)
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def bounded_joint_acc_l2(
  env: ManagerBasedRlEnv,
  max_penalty: float = 1_000_000.0,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  penalty = base_envs_mdp.joint_acc_l2(env, asset_cfg=asset_cfg)
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def _torso_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].amax(dim=1)
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    torso_height = torso_height - env_origins[:, 2]
  return torso_height


def getup_posture_reward(
  env: ManagerBasedRlEnv,
  tilt_std: float = 0.35,
  torso_height_target: float = 0.62,
  torso_height_std: float = 0.2,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  upright_alignment = torch.clamp(_upright_alignment(projected_gravity_b), min=0.0, max=1.0)
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  tilt_term = torch.exp(-torch.square(tilt) / max(tilt_std**2, 1e-6))
  height_term = torch.exp(-torch.square(torso_height - torso_height_target) / max(torso_height_std**2, 1e-6))
  return tilt_term * height_term * upright_alignment


def getup_height_progress_reward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.18,
  target_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b
  upright_alignment = torch.clamp(_upright_alignment(projected_gravity_b), min=0.0, max=1.0)
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  return progress * upright_alignment


def getup_torso_lift_reward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.12,
  target_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  return torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )


def getup_facing_up_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)


class getup_phase_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._paid_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    thresholds: tuple[float, ...] = (0.22, 0.4, 0.55),
    bonuses: tuple[float, ...] = (1.0, 2.0, 3.0),
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    assert len(thresholds) == len(bonuses), "thresholds and bonuses must align"
    torso_height = _torso_height(env, asset_cfg=asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    for stage_idx, (threshold, bonus) in enumerate(zip(thresholds, bonuses, strict=False), start=1):
      newly_crossed = (self._paid_stage < stage_idx) & (torso_height >= threshold)
      reward = reward + newly_crossed.float() * bonus
      self._paid_stage[newly_crossed] = stage_idx
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid_stage[env_ids] = 0


class getup_orientation_phase_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._paid_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    thresholds: tuple[float, ...] = (0.1, 0.4, 0.7),
    bonuses: tuple[float, ...] = (1.0, 2.0, 3.0),
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    assert len(thresholds) == len(bonuses), "thresholds and bonuses must align"
    asset: Entity = env.scene[asset_cfg.name]
    alignment = torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)
    reward = torch.zeros(env.num_envs, device=env.device)
    for stage_idx, (threshold, bonus) in enumerate(zip(thresholds, bonuses, strict=False), start=1):
      newly_crossed = (self._paid_stage < stage_idx) & (alignment >= threshold)
      reward = reward + newly_crossed.float() * bonus
      self._paid_stage[newly_crossed] = stage_idx
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid_stage[env_ids] = 0


class reduced_support_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_support_count: float = 0.5,
    activation_height: float = 0.4,
    alignment_threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    support_count = (sensor_data.found > 0).float().sum(dim=1)
    asset: Entity = env.scene[asset_cfg.name]
    torso_height = _torso_height(env, asset_cfg=asset_cfg)
    facing_up = _upright_alignment(asset.data.projected_gravity_b) >= alignment_threshold
    eligible = (torso_height >= activation_height) & facing_up & (support_count <= max_support_count) & ~self._paid
    reward = eligible.float()
    self._paid[eligible] = True
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid[env_ids] = False


class getup_demo_pose_reward:
  def __init__(
    self,
    cfg,
    env: ManagerBasedRlEnv,
    demo_npz_path: str | None = None,
    joint_names: tuple[str, ...] | None = None,
    dt_per_demo_frame: float | None = None,
  ):
    params = getattr(cfg, "params", {}) or {}
    if demo_npz_path is None:
      demo_npz_path = params.get("demo_npz_path")
    if joint_names is None:
      joint_names = params.get("joint_names")
    if dt_per_demo_frame is None:
      dt_per_demo_frame = params.get("dt_per_demo_frame", 0.02)
    if demo_npz_path is None or joint_names is None:
      raise TypeError("getup_demo_pose_reward requires demo_npz_path and joint_names")
    self._elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    payload = np.load(Path(demo_npz_path).expanduser())
    joint_pos = torch.tensor(payload["joint_pos"], dtype=torch.float32, device=env.device)
    self._demo_joint_pos = joint_pos
    asset: Entity = env.scene[_JOINT_ASSET_CFG.name]
    joint_name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    self._joint_ids = torch.tensor(
      [joint_name_to_index[name] for name in joint_names if name in joint_name_to_index],
      dtype=torch.long,
      device=env.device,
    )
    step_dt = getattr(env, "step_dt", dt_per_demo_frame)
    self._frames_per_step = max(1, int(round(step_dt / max(dt_per_demo_frame, 1e-6))))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    demo_npz_path: str,
    joint_names: tuple[str, ...],
    dt_per_demo_frame: float = 0.02,
    std: float = 0.75,
  ) -> torch.Tensor:
    del demo_npz_path, joint_names, dt_per_demo_frame
    frame_idx = torch.clamp(self._elapsed_steps * self._frames_per_step, max=self._demo_joint_pos.shape[0] - 1)
    target = self._demo_joint_pos[frame_idx][:, self._joint_ids]
    asset: Entity = env.scene[_JOINT_ASSET_CFG.name]
    current = asset.data.joint_pos[:, self._joint_ids]
    reward = torch.exp(-torch.mean(torch.square(current - target), dim=1) / max(std**2, 1e-6))
    self._elapsed_steps += 1
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._elapsed_steps[env_ids] = 0


def stand_still_after_getup(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  activation_height: float = 0.45,
  facing_up_threshold: float = 0.3,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  joint_asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[torso_asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=torso_asset_cfg)
  facing_up = _upright_alignment(asset.data.projected_gravity_b) >= facing_up_threshold
  active = (torso_height >= activation_height) & facing_up
  penalty = _base_stand_still(
    env,
    command_name=command_name,
    command_threshold=command_threshold,
    asset_cfg=joint_asset_cfg,
  )
  return penalty * active.float()


def action_rate_after_lift(
  env: ManagerBasedRlEnv,
  activation_height: float = 0.25,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  active = (torso_height >= activation_height).float()
  penalty = envs_mdp.action_rate_l2(env)
  return penalty * active


def bounded_action_rate_after_lift(
  env: ManagerBasedRlEnv,
  activation_height: float = 0.25,
  max_penalty: float = 250.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  active = (torso_height >= activation_height).float()
  penalty = envs_mdp.action_rate_l2(env)
  penalty = _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)
  return penalty * active


def track_linear_velocity_after_lift(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  activation_height: float = 0.45,
  alignment_threshold: float = 0.3,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  facing_up = _upright_alignment(asset.data.projected_gravity_b) >= alignment_threshold
  active = (torso_height >= activation_height) & facing_up
  reward = _base_track_linear_velocity(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
  return reward * active.float()


def track_angular_velocity_after_lift(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  activation_height: float = 0.45,
  alignment_threshold: float = 0.3,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  facing_up = _upright_alignment(asset.data.projected_gravity_b) >= alignment_threshold
  active = (torso_height >= activation_height) & facing_up
  reward = _base_track_angular_velocity(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
  return reward * active.float()


def support_contact_diversity_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_count: float = 2.0,
  active_below_height: float | None = None,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  contact_count = (sensor_data.found > 0).float().sum(dim=1)
  reward = torch.exp(-torch.square(contact_count - target_count))
  if active_below_height is not None:
    torso_height = _torso_height(env, asset_cfg=asset_cfg)
    reward = reward * (torso_height < active_below_height).float()
  return reward


def support_body_contact_penalty_after_lift(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  activation_height: float = 0.35,
  normalize_count: float = 2.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  contact_count = (sensor_data.found > 0).float().sum(dim=1)
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  active = (torso_height >= activation_height).float()
  normalized = torch.clamp(contact_count / max(normalize_count, 1e-6), min=0.0, max=1.0)
  return normalized * active


def getup_feet_support_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  max_body_support_count: float = 4.0,
  alignment_floor: float = -0.1,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  feet_sensor = env.scene[feet_sensor_name]
  feet_sensor_data = feet_sensor.data
  assert feet_sensor_data.found is not None
  feet_count = (feet_sensor_data.found > 0).float().sum(dim=1)
  feet_support = torch.clamp(feet_count / 2.0, min=0.0, max=1.0)

  body_sensor = env.scene[body_sensor_name]
  body_sensor_data = body_sensor.data
  assert body_sensor_data.found is not None
  body_count = (body_sensor_data.found > 0).float().sum(dim=1)
  body_relief = 1.0 - torch.clamp(body_count / max(max_body_support_count, 1e-6), min=0.0, max=1.0)

  asset: Entity = env.scene[asset_cfg.name]
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  alignment_progress = torch.clamp(
    (alignment - alignment_floor) / max(1.0 - alignment_floor, 1e-6),
    min=0.0,
    max=1.0,
  )
  return feet_support * (0.25 + 0.75 * body_relief) * (0.25 + 0.75 * alignment_progress)


def getup_standing_joint_pose_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  joint_names: tuple[str, ...],
  std: float = 0.75,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 2.0,
  alignment_threshold: float = 0.0,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
  selected_joint_ids = [joint_name_to_index[name] for name in joint_names if name in joint_name_to_index]
  if not selected_joint_ids:
    return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)

  selected_joint_ids_t = torch.tensor(selected_joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
  joint_error = asset.data.joint_pos[:, selected_joint_ids_t] - asset.data.default_joint_pos[:, selected_joint_ids_t]
  pose_reward = torch.exp(-torch.mean(torch.square(joint_error), dim=1) / max(std**2, 1e-6))

  feet_sensor = env.scene[feet_sensor_name]
  feet_found = feet_sensor.data.found
  assert feet_found is not None
  feet_contact_count = (feet_found > 0).float().sum(dim=1)

  body_sensor = env.scene[body_sensor_name]
  body_found = body_sensor.data.found
  assert body_found is not None
  body_support_count = (body_found > 0).float().sum(dim=1)

  alignment = _upright_alignment(asset.data.projected_gravity_b)
  active = (
    (feet_contact_count >= min_feet_contact_count)
    & (body_support_count <= max_body_support_count)
    & (alignment >= alignment_threshold)
  )
  return pose_reward * active.float()


def joint_pos_limits_after_support(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 2.0,
  alignment_threshold: float = 0.0,
  max_penalty: float = 10.0,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  feet_sensor = env.scene[feet_sensor_name]
  feet_found = feet_sensor.data.found
  assert feet_found is not None
  feet_contact_count = (feet_found > 0).float().sum(dim=1)

  body_sensor = env.scene[body_sensor_name]
  body_found = body_sensor.data.found
  assert body_found is not None
  body_support_count = (body_found > 0).float().sum(dim=1)

  asset: Entity = env.scene[asset_cfg.name]
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  active = (
    (feet_contact_count >= min_feet_contact_count)
    & (body_support_count <= max_body_support_count)
    & (alignment >= alignment_threshold)
  )
  penalty = base_envs_mdp.joint_pos_limits(env, asset_cfg=asset_cfg)
  penalty = _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)
  return penalty * active.float()


def self_collision_cost_after_support(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_sensor_name: str,
  body_sensor_name: str,
  force_threshold: float = 10.0,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 2.0,
  alignment_threshold: float = 0.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  feet_sensor = env.scene[feet_sensor_name]
  feet_found = feet_sensor.data.found
  assert feet_found is not None
  feet_contact_count = (feet_found > 0).float().sum(dim=1)

  body_sensor = env.scene[body_sensor_name]
  body_found = body_sensor.data.found
  assert body_found is not None
  body_support_count = (body_found > 0).float().sum(dim=1)

  asset: Entity = env.scene[asset_cfg.name]
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  active = (
    (feet_contact_count >= min_feet_contact_count)
    & (body_support_count <= max_body_support_count)
    & (alignment >= alignment_threshold)
  )
  return _base_self_collision_cost(env, sensor_name=sensor_name, force_threshold=force_threshold) * active.float()


def pelvis_clearance_penalty(
  env: ManagerBasedRlEnv,
  min_clearance: float = 0.05,
  penalty_scale: float = 2.0,
) -> torch.Tensor:
  violation = pelvis_clearance_violation(env, min_clearance=min_clearance)
  return violation * penalty_scale


def getup_completion_bonus(
  env: ManagerBasedRlEnv,
  tilt_threshold: float = 0.3,
  torso_height_threshold: float = 0.55,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  return getup_upright(
    env,
    tilt_threshold=tilt_threshold,
    torso_height_threshold=torso_height_threshold,
    asset_cfg=asset_cfg,
  )
