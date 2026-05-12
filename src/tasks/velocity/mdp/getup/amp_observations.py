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


def _yaw_invariant_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Remove world-frame yaw so demo and policy share the same heading frame.

  AMP must be yaw-invariant: the demo character walks across the world while
  the policy starts near the spawn origin.  Without yaw normalization the
  discriminator separates trajectories by absolute heading and the AMP reward
  saturates instantly.
  """
  w, x, y, z = quat_wxyz.unbind(dim=-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  half = -0.5 * yaw
  yaw_w = torch.cos(half)
  yaw_z = torch.sin(half)
  new_w = yaw_w * w - yaw_z * z
  new_x = yaw_w * x - yaw_z * y
  new_y = yaw_w * y + yaw_z * x
  new_z = yaw_w * z + yaw_z * w
  out = torch.stack([new_w, new_x, new_y, new_z], dim=-1)
  return out / torch.clamp(torch.linalg.norm(out, dim=-1, keepdim=True), min=1e-6)


def amp_root_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Z component of root link relative to env origin."""
  asset = _robot(env, asset_cfg)
  root_pos = asset.data.root_link_pos_w
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    root_pos = root_pos - env_origins
  return root_pos[:, 2:3]


def amp_root_quat_yaw_invariant(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  return _yaw_invariant_quat_wxyz(asset.data.root_link_quat_w)


def amp_joint_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  joint_ids = asset_cfg.joint_ids
  return asset.data.joint_pos[:, joint_ids]


def amp_joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset = _robot(env, asset_cfg)
  joint_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, joint_ids]


def amp_getup_features(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """Concatenate env AMP features matching standardized demo features.

  Drops the world-frame XY position and removes yaw from root_quat so that
  policy and demo distributions overlap on a heading-invariant frame.
  Resulting layout: [root_z (1), root_quat_no_yaw (4), joint_pos (23), joint_vel (23)] = 51 dims.
  """
  return torch.cat(
    [
      amp_root_height(env, asset_cfg=asset_cfg),
      amp_root_quat_yaw_invariant(env, asset_cfg=asset_cfg),
      amp_joint_pos(env, asset_cfg=asset_cfg),
      amp_joint_vel(env, asset_cfg=asset_cfg),
    ],
    dim=-1,
  )
