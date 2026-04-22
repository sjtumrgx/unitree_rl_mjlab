from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .metrics import getup_upright, pelvis_clearance_violation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_TORSO_ASSET_CFG = SceneEntityCfg("robot", body_names=("torso_link",))


def getup_posture_reward(
  env: ManagerBasedRlEnv,
  tilt_std: float = 0.35,
  torso_height_target: float = 0.62,
  torso_height_std: float = 0.2,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].amax(dim=1)
  tilt_term = torch.exp(-torch.square(tilt) / max(tilt_std**2, 1e-6))
  height_term = torch.exp(-torch.square(torso_height - torso_height_target) / max(torso_height_std**2, 1e-6))
  return tilt_term * height_term


def support_contact_diversity_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_count: float = 2.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  contact_count = (sensor_data.found > 0).float().sum(dim=1)
  return torch.exp(-torch.square(contact_count - target_count))


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
