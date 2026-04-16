from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .events import (
  DISTURBANCE_NEAR_FAILURE_RESET,
  disturbance_age_fraction,
  disturbance_window_mask,
  get_antifall_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def disturbance_metadata(
  env: ManagerBasedRlEnv,
  window_s: float = 2.0,
  magnitude_scale: float = 4.0,
) -> torch.Tensor:
  state = get_antifall_state(env)
  magnitude = torch.clamp(
    state["last_disturbance_mag"] / max(magnitude_scale, 1e-6),
    0.0,
    1.0,
  )
  kind = state["disturbance_kind"].float() / float(DISTURBANCE_NEAR_FAILURE_RESET)
  return torch.stack(
    [
      state["disturbance_active"].float(),
      magnitude,
      disturbance_age_fraction(env, window_s),
      kind,
    ],
    dim=1,
  )


def recovery_features(
  env: ManagerBasedRlEnv,
  command_name: str,
  window_s: float = 2.0,
  tracking_threshold: float = 0.5,
  yaw_threshold: float = 0.75,
  tilt_threshold: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  lin_error = torch.linalg.norm(
    command[:, :2] - asset.data.root_link_lin_vel_b[:, :2], dim=1
  )
  yaw_error = torch.abs(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  window = disturbance_window_mask(env, window_s).float()
  controllable = (
    (lin_error <= tracking_threshold)
    & (yaw_error <= yaw_threshold)
    & (tilt <= tilt_threshold)
  ).float()
  return torch.stack(
    [
      window,
      disturbance_age_fraction(env, window_s),
      controllable,
      torch.clamp(1.0 - tilt / max(tilt_threshold, 1e-6), 0.0, 1.0),
    ],
    dim=1,
  )
