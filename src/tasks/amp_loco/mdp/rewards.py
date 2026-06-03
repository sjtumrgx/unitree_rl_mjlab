from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  yaw_quat,
  quat_apply
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_delay_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Get delaying env mask from DelayedTerminationManager if installed."""
  tm = env.termination_manager
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  delay_counters = getattr(tm, "_delay_counters", None)
  if isinstance(delay_env_mask, torch.Tensor) and isinstance(delay_counters, torch.Tensor):
    return delay_env_mask & (delay_counters > 0)
  return None


def _apply_delay_env_reward_scaling(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return reward

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return reward

  scaled_reward = reward * delay_env_rew_ratio
  return torch.where(delay_env_mask, scaled_reward, reward)


def _apply_delay_env_reward_mask_only(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return torch.zeros_like(reward)

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return torch.zeros_like(reward)

  scaled_reward = reward * delay_env_rew_ratio
  masked_reward = torch.where(delay_env_mask, scaled_reward, torch.zeros_like(reward))
  return masked_reward

def track_anchor_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward for tracking the commanded anchor linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  command_xyz_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
  command_xyz_w = quat_apply(
    yaw_quat(asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]),
    command_xyz_b,
  )
  lin_vel_error = torch.sum(torch.square(command_xyz_w[:,:3] - asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]), dim=1)
  reward = torch.exp(-lin_vel_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  anchor_ang_vel_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
  anchor_ang_z_vel_w = anchor_ang_vel_w[:, 2]
  command_ang_vel_w = command[:, 2]
  ang_vel_z_error = torch.square(command_ang_vel_w - anchor_ang_z_vel_w)

  anchor_ang_vel_b =  quat_apply_inverse(
    asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]],
    anchor_ang_vel_w,
  )
  ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)

  total_error = ang_vel_z_error + ang_vel_xy_error

  reward = torch.exp(-total_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)

def body_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[body_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids[0]]
  body_ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, body_cfg.body_ids[0]],
    body_ang_vel_w,
  )
  body_ang_vel_xy_b = body_ang_vel_b[:, :2]
  ang_vel_xy_error = torch.sum(torch.square(body_ang_vel_xy_b), dim=-1)

  reward = torch.exp(-ang_vel_xy_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)

def track_root_height(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded anchor height."""
  asset: Entity = env.scene[asset_cfg.name]

  desired_height = asset.data.default_root_state[:, 2]
  cur_root_height = asset.data.body_link_pos_w[:, 0, 2]
  height_error = torch.square(desired_height - cur_root_height)
  reward = torch.exp(-height_error / std**2)
  return _apply_delay_env_reward_mask_only(env, reward, mask_delay, delay_env_rew_ratio)

def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost

def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost

def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
