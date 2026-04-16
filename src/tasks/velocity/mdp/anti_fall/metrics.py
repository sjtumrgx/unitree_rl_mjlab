from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .events import disturbance_age_s, disturbance_window_mask, get_antifall_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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


def disturbance_window_active(
  env: ManagerBasedRlEnv,
  window_s: float = 2.0,
) -> torch.Tensor:
  return disturbance_window_mask(env, window_s).float()


def disturbance_magnitude(
  env: ManagerBasedRlEnv,
  magnitude_scale: float = 4.0,
) -> torch.Tensor:
  state = get_antifall_state(env)
  return torch.clamp(
    state["last_disturbance_mag"] / max(magnitude_scale, 1e-6),
    0.0,
    1.0,
  )


def controllable_locomotion(
  env: ManagerBasedRlEnv,
  command_name: str,
  tracking_threshold: float = 0.5,
  yaw_threshold: float = 0.75,
  tilt_threshold: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return _controllable_mask(
    env,
    command_name=command_name,
    tracking_threshold=tracking_threshold,
    yaw_threshold=yaw_threshold,
    tilt_threshold=tilt_threshold,
    asset_cfg=asset_cfg,
  ).float()


class disturbance_count:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._last_seen = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    state = get_antifall_state(env)
    delta = torch.clamp(state["disturbance_count"] - self._last_seen, min=0)
    self._last_seen[:] = state["disturbance_count"]
    return delta.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last_seen[env_ids] = 0


class recovery_success_count:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._last_success = torch.full(
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
    success = disturbance_window_mask(env, window_s)
    success &= disturbance_age_s(env) >= min_recovery_delay_s
    success &= _controllable_mask(
      env,
      command_name=command_name,
      tracking_threshold=tracking_threshold,
      yaw_threshold=yaw_threshold,
      tilt_threshold=tilt_threshold,
      asset_cfg=asset_cfg,
    )
    success &= state["disturbance_count"] > 0
    success &= self._last_success != state["disturbance_count"]
    out = success.float()
    self._last_success[success] = state["disturbance_count"][success]
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last_success[env_ids] = -1


class recovery_latency:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._last_reported = torch.full(
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
    age_s = disturbance_age_s(env)
    ready = disturbance_window_mask(env, window_s)
    ready &= age_s >= min_recovery_delay_s
    ready &= _controllable_mask(
      env,
      command_name=command_name,
      tracking_threshold=tracking_threshold,
      yaw_threshold=yaw_threshold,
      tilt_threshold=tilt_threshold,
      asset_cfg=asset_cfg,
    )
    ready &= state["disturbance_count"] > 0
    ready &= self._last_reported != state["disturbance_count"]
    out = torch.zeros(env.num_envs, device=env.device)
    out[ready] = age_s[ready]
    self._last_reported[ready] = state["disturbance_count"][ready]
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._last_reported[env_ids] = -1
