from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..rewards import track_angular_velocity, track_linear_velocity
from .events import disturbance_window_mask, get_antifall_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _select_joint_view(
  joint_tensor: torch.Tensor,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  joint_ids = getattr(asset_cfg, "joint_ids", None)
  if joint_ids is None:
    return joint_tensor
  if isinstance(joint_ids, slice):
    return joint_tensor[:, joint_ids]
  if len(joint_ids) == 0:
    return joint_tensor
  return joint_tensor[:, joint_ids]


def _controllable_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_threshold: float,
  yaw_threshold: float,
  tilt_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  lin_error = torch.linalg.norm(
    command[:, :2] - asset.data.root_link_lin_vel_b[:, :2], dim=1
  )
  yaw_error = torch.abs(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  return (
    (lin_error <= tracking_threshold)
    & (yaw_error <= yaw_threshold)
    & (tilt <= tilt_threshold)
  )


def _log_scalar(env: ManagerBasedRlEnv, key: str, value: torch.Tensor) -> None:
  log = env.extras.setdefault("log", {})
  log[key] = torch.mean(value)


def upright_recoverability(
  env: ManagerBasedRlEnv,
  tilt_std: float = 0.35,
  ang_vel_std: float = 1.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  tilt_cost = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  ang_vel_cost = torch.sum(
    torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1
  )
  reward = torch.exp(
    -tilt_cost / max(tilt_std**2, 1e-6)
    - ang_vel_cost / max(ang_vel_std**2, 1e-6)
  )
  _log_scalar(env, "Metrics/upright_recoverability", reward)
  return reward


def recovery_quality(
  env: ManagerBasedRlEnv,
  command_name: str,
  window_s: float = 2.0,
  lin_vel_std: float = 0.5,
  ang_vel_std: float = 0.75,
  tilt_std: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  window = disturbance_window_mask(env, window_s).float()
  if not window.any():
    return torch.zeros(env.num_envs, device=env.device)
  reward = track_linear_velocity(
    env,
    std=lin_vel_std,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  reward *= track_angular_velocity(
    env,
    std=ang_vel_std,
    command_name=command_name,
    asset_cfg=asset_cfg,
  )
  reward *= upright_recoverability(
    env,
    tilt_std=tilt_std,
    ang_vel_std=ang_vel_std,
    asset_cfg=asset_cfg,
  )
  reward *= window
  _log_scalar(env, "Metrics/recovery_quality", reward)
  return reward


def standing_stability(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  joint_vel_std: float = 3.0,
  ang_vel_std: float = 1.5,
  tilt_std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  command_norm = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  is_standing = (command_norm <= command_threshold).float()
  joint_vel = _select_joint_view(asset.data.joint_vel, asset_cfg)
  joint_vel_cost = torch.mean(torch.square(joint_vel), dim=1)
  ang_vel_cost = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  tilt_cost = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  reward = torch.exp(
    -joint_vel_cost / max(joint_vel_std**2, 1e-6)
    - ang_vel_cost / max(ang_vel_std**2, 1e-6)
    - tilt_cost / max(tilt_std**2, 1e-6)
  )
  reward *= is_standing
  _log_scalar(env, "Metrics/standing_stability", reward)
  return reward


class recovery_completion_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._paid_count = torch.full(
      (env.num_envs,),
      -1,
      dtype=torch.long,
      device=env.device,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    window_s: float = 2.0,
    tracking_threshold: float = 0.5,
    yaw_threshold: float = 0.75,
    tilt_threshold: float = 0.35,
    min_recovery_delay_s: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    state = get_antifall_state(env)
    age_s = torch.clamp(
      (int(getattr(env, "common_step_counter", 0)) - state["last_disturbance_step"]).float()
      * env.step_dt,
      min=0.0,
    )
    window = disturbance_window_mask(env, window_s)
    controllable = _controllable_mask(
      env,
      command_name=command_name,
      tracking_threshold=tracking_threshold,
      yaw_threshold=yaw_threshold,
      tilt_threshold=tilt_threshold,
      asset_cfg=asset_cfg,
    )
    eligible = (
      window
      & (age_s >= min_recovery_delay_s)
      & (state["disturbance_count"] > 0)
      & controllable
      & (self._paid_count != state["disturbance_count"])
    )
    reward = eligible.float()
    self._paid_count[eligible] = state["disturbance_count"][eligible]
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid_count[env_ids] = -1
