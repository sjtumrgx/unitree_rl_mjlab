from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_TORSO_ASSET_CFG = SceneEntityCfg("robot", body_names=("torso_link",))
_PELVIS_ASSET_CFG = SceneEntityCfg("robot", body_names=("pelvis",))


def _upright_alignment(projected_gravity_b: torch.Tensor) -> torch.Tensor:
  return -projected_gravity_b[:, 2]


def _is_facing_up(projected_gravity_b: torch.Tensor, tilt_threshold: float) -> torch.Tensor:
  min_up_alignment = math.cos(tilt_threshold)
  return _upright_alignment(projected_gravity_b) >= min_up_alignment


def _relative_body_height(
  env: ManagerBasedRlEnv,
  body_heights: torch.Tensor,
) -> torch.Tensor:
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    body_heights = body_heights - env_origins[:, 2].unsqueeze(1)
  return body_heights




def support_body_contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float().sum(dim=1)


def torso_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("pelvis", "torso_link")),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  clearance = _relative_body_height(env, asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2])
  return clearance.min(dim=1).values

def getup_upright(
  env: ManagerBasedRlEnv,
  tilt_threshold: float = 0.3,
  torso_height_threshold: float = 0.55,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _relative_body_height(env, asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]).amax(dim=1)
  projected_gravity_b = asset.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  facing_up = _is_facing_up(projected_gravity_b, tilt_threshold=tilt_threshold)
  return ((tilt <= tilt_threshold) & facing_up & (torso_height >= torso_height_threshold)).float()


def pelvis_clearance_violation(
  env: ManagerBasedRlEnv,
  min_clearance: float = 0.05,
  asset_cfg: SceneEntityCfg = _PELVIS_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  pelvis_height = _relative_body_height(env, asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]).amin(dim=1)
  return (pelvis_height < min_clearance).float()


class getup_success_count:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._reported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    tilt_threshold: float = 0.3,
    torso_height_threshold: float = 0.55,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    success = getup_upright(
      env,
      tilt_threshold=tilt_threshold,
      torso_height_threshold=torso_height_threshold,
      asset_cfg=asset_cfg,
    ).bool()
    new_success = success & ~self._reported
    self._reported[new_success] = True
    return new_success.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._reported[env_ids] = False


class getup_latency:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._reported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    tilt_threshold: float = 0.3,
    torso_height_threshold: float = 0.55,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    success = getup_upright(
      env,
      tilt_threshold=tilt_threshold,
      torso_height_threshold=torso_height_threshold,
      asset_cfg=asset_cfg,
    ).bool()
    new_success = success & ~self._reported
    out = torch.zeros(env.num_envs, device=env.device)
    out[new_success] = env.episode_length_buf[new_success].float() * env.step_dt
    self._reported[new_success] = True
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._reported[env_ids] = False
