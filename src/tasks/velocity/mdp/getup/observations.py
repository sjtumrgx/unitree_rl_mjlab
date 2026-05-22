from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

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
  contact_count = (sensor_data.found > 0).float().flatten(start_dim=1).sum(dim=1)
  return torch.clamp(contact_count / max(normalize_count, 1e-6), min=0.0, max=1.0)


def _env_origins(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  return env_origins


def _body_lin_vel_w(asset: Entity) -> torch.Tensor:
  vel = getattr(asset.data, "body_link_lin_vel_w", None)
  if vel is not None:
    return vel
  vel = getattr(asset.data, "body_link_vel_w", None)
  if vel is not None:
    return vel
  raise AttributeError("robot data must expose body_link_lin_vel_w or body_link_vel_w")


def _quat_tan_norm_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """BFM-style 6D rotation observation from local wxyz quaternions.

  BFM-Zero's ``max_local_self`` represents each body orientation by rotating a
  tangent x-axis and a normal z-axis.  Keep the same information layout while
  using MJLab's wxyz quaternion convention.
  """

  ref_tan = torch.zeros_like(quat_wxyz[..., :3])
  ref_tan[..., 0] = 1.0
  tan = quat_apply(quat_wxyz, ref_tan)

  ref_norm = torch.zeros_like(quat_wxyz[..., :3])
  ref_norm[..., 2] = 1.0
  norm = quat_apply(quat_wxyz, ref_norm)
  return torch.cat([tan, norm], dim=-1)


def support_body_contact_pattern(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float().flatten(start_dim=1)


def bfm_local_body_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  root_height_obs: bool = True,
  local_root_obs: bool = True,
) -> torch.Tensor:
  """BFM-Zero style heading-invariant full-body self state for GetUp.

  The previous GetUp actor only observed IMU-like base terms, joints, actions,
  and a few scalar progress/contact features.  BFM-Zero's fall-recovery policy
  exposes a much richer ``max_local_self`` signal: root height plus each body's
  root-relative position, heading-local orientation, linear velocity, and
  angular velocity.  This term transfers that observation contract to MJLab so
  a no-assist policy can reason about where the torso/limbs are while fallen.

  Layout:
    [root_height? (1),
     yaw-invariant body positions relative to root, with root body dropped,
     yaw-invariant body rotations as tan+norm 6D for every body,
     yaw-invariant body linear velocities for every body,
     yaw-invariant body angular velocities for every body]
  """

  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  body_pos = asset.data.body_link_pos_w[:, body_ids, :]
  body_quat = asset.data.body_link_quat_w[:, body_ids, :]
  body_vel = _body_lin_vel_w(asset)[:, body_ids, :]
  body_ang_vel = asset.data.body_link_ang_vel_w[:, body_ids, :]

  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  env_origins = _env_origins(env)
  if env_origins is not None:
    root_height = root_pos[:, 2:3] - env_origins[:, 2:3]
  else:
    root_height = root_pos[:, 2:3]

  heading_quat = yaw_quat(root_quat)
  num_bodies = int(body_pos.shape[1])
  heading_quat_body = heading_quat[:, None, :].expand(-1, num_bodies, -1)

  local_body_pos = body_pos - root_pos[:, None, :]
  local_body_pos = quat_apply_inverse(heading_quat_body, local_body_pos)
  local_body_pos = local_body_pos.reshape(local_body_pos.shape[0], -1)
  if local_body_pos.shape[1] >= 3:
    local_body_pos = local_body_pos[:, 3:]

  inv_heading = heading_quat.clone()
  inv_heading[:, 1:] = -inv_heading[:, 1:]
  inv_heading_body = inv_heading[:, None, :].expand(-1, num_bodies, -1)
  local_body_quat = quat_mul(inv_heading_body.reshape(-1, 4), body_quat.reshape(-1, 4))
  local_body_rot = _quat_tan_norm_wxyz(local_body_quat).reshape(body_quat.shape[0], -1)
  if not local_root_obs and local_body_rot.shape[1] >= 6:
    local_body_rot[:, :6] = _quat_tan_norm_wxyz(root_quat)

  local_body_vel = quat_apply_inverse(heading_quat_body, body_vel).reshape(body_vel.shape[0], -1)
  local_body_ang_vel = quat_apply_inverse(heading_quat_body, body_ang_vel).reshape(body_ang_vel.shape[0], -1)

  features = []
  if root_height_obs:
    features.append(root_height)
  features.extend([local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel])
  return torch.nan_to_num(torch.cat(features, dim=-1), nan=0.0, posinf=1e6, neginf=-1e6)


def torso_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  env_origins = _env_origins(env)
  if env_origins is not None:
    torso_height = torso_height - env_origins[:, None, 2]
  return torso_height


def getup_progress_features(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_sensor_name: str | None = None,
  hand_sensor_name: str | None = None,
  min_height: float = 0.12,
  target_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].amax(dim=1)
  env_origins = _env_origins(env)
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
  hand_support_norm = _contact_norm(env, hand_sensor_name, normalize_count=2.0)
  features = (height_progress, facing_up, body_support_norm)
  if feet_sensor_name is not None:
    features = (*features, feet_support_norm)
  if hand_sensor_name is not None:
    features = (*features, hand_support_norm)
  return torch.stack(features, dim=1)


def host_effective_actions(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  """Return HoST get-up actions after unactuated warmup/curriculum gating.

  MJLab's generic ``last_action`` observation exposes raw policy output.  For
  HoST get-up, the first warmup steps intentionally do not execute policy
  deltas, and curriculum scaling/clamping further changes the applied delta.
  Actor/critic observations should therefore see the effective action that was
  actually sent to the actuators, not a raw command that never took effect.
  """

  action_manager = getattr(env, "action_manager", None)
  if action_manager is not None:
    try:
      term = action_manager.get_term(action_name)
    except (AttributeError, KeyError):
      term = None
    effective_action = getattr(term, "effective_action", None)
    if effective_action is not None:
      return effective_action

  effective_action = getattr(env, "_host_getup_effective_action", None)
  if effective_action is not None:
    return effective_action

  if action_manager is None:
    return torch.zeros(_batch_size(env), 0, device=getattr(env, "device", None) or "cpu")
  return action_manager.action
