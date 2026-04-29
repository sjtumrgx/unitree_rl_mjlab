"""AMP discriminator observation terms for the optional G1 GetUp fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def amp_root_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  root_pos = asset.data.root_link_pos_w
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    root_pos = root_pos - env_origins
  return root_pos


def amp_root_quat_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return asset.data.root_link_quat_w


def amp_joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  joint_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, joint_ids]


def amp_joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  joint_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, joint_ids]


def amp_getup_features(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Concatenate env AMP features in the same order as standardized demos."""
  return torch.cat(
    [
      amp_root_pos_w(env, asset_cfg=asset_cfg),
      amp_root_quat_w(env, asset_cfg=asset_cfg),
      amp_joint_pos(env, asset_cfg=asset_cfg),
      amp_joint_vel(env, asset_cfg=asset_cfg),
    ],
    dim=-1,
  )
