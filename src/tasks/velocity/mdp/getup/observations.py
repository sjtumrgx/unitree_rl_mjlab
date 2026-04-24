from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .metrics import _upright_alignment

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _batch_size(env: ManagerBasedRlEnv) -> int:
  if hasattr(env, "num_envs"):
    return int(env.num_envs)
  robot = env.scene["robot"]
  return int(robot.data.projected_gravity_b.shape[0])


def _contact_norm(
  env: ManagerBasedRlEnv,
  sensor_name: str | None,
  *,
  normalize_count: float,
) -> torch.Tensor:
  if sensor_name is None:
    return torch.zeros(_batch_size(env), device=getattr(env, "device", None) or "cpu")
  sensor = env.scene.get(sensor_name) if isinstance(env.scene, dict) else env.scene[sensor_name]
  if sensor is None:
    return torch.zeros(_batch_size(env), device=getattr(env, "device", None) or "cpu")
  sensor_data = sensor.data
  assert sensor_data.found is not None
  contact_count = (sensor_data.found > 0).float().sum(dim=1)
  return torch.clamp(contact_count / max(normalize_count, 1e-6), min=0.0, max=1.0)


def support_body_contact_pattern(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float().flatten(start_dim=1)


def torso_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    torso_height = torso_height - env_origins[:, None, 2]
  return torso_height


def getup_progress_features(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_sensor_name: str | None = None,
  min_height: float = 0.12,
  target_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].amax(dim=1)
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    torso_height = torso_height - env_origins[:, 2]
  height_progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  facing_up = torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)
  body_support_norm = _contact_norm(env, sensor_name, normalize_count=2.0)
  feet_support_norm = _contact_norm(env, feet_sensor_name, normalize_count=2.0)
  features = (height_progress, facing_up, body_support_norm)
  if feet_sensor_name is not None:
    features = (*features, feet_support_norm)
  return torch.stack(features, dim=1)
