from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import torch
import numpy as np

from mjlab.envs import mdp as envs_mdp
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp as base_envs_mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, wrap_to_pi

from ..rewards import (
  angular_momentum_penalty as _base_angular_momentum_penalty,
  body_angular_velocity_penalty as _base_body_angular_velocity_penalty,
  stand_still as _base_stand_still,
  track_angular_velocity as _base_track_angular_velocity,
  track_linear_velocity as _base_track_linear_velocity,
  self_collision_cost as _base_self_collision_cost,
)
from .metrics import _upright_alignment, getup_upright, pelvis_clearance_violation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_TORSO_ASSET_CFG = SceneEntityCfg("robot", body_names=("torso_link",))
_JOINT_ASSET_CFG = SceneEntityCfg("robot")
_FOOT_ASSET_CFG = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))


def _bounded_nonnegative_penalty(
  value: torch.Tensor,
  *,
  max_penalty: float,
) -> torch.Tensor:
  """Clamp positive penalty signals before they dominate PPO returns."""

  max_value = float(max_penalty)
  finite = torch.nan_to_num(value, nan=max_value, posinf=max_value, neginf=max_value)
  return torch.clamp(finite, min=0.0, max=max_value)


def bounded_body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  max_penalty: float = 400.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  penalty = _base_body_angular_velocity_penalty(env, asset_cfg=asset_cfg)
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def bounded_angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_penalty: float = 1000.0,
) -> torch.Tensor:
  env.extras.setdefault("log", {})
  penalty = _base_angular_momentum_penalty(env, sensor_name=sensor_name)
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def bounded_joint_acc_l2(
  env: ManagerBasedRlEnv,
  max_penalty: float = 1_000_000.0,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  penalty = base_envs_mdp.joint_acc_l2(env, asset_cfg=asset_cfg)
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def _torso_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].amax(dim=1)
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    torso_height = torso_height - env_origins[:, 2]
  return torso_height


def _contact_count(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found > 0).float().flatten(start_dim=1).sum(dim=1)


def _contact_norm(env: ManagerBasedRlEnv, sensor_name: str | None, normalize_count: float) -> torch.Tensor:
  if sensor_name is None:
    return torch.zeros(env.num_envs, device=env.device)
  sensor = env.scene.get(sensor_name) if isinstance(env.scene, dict) else env.scene[sensor_name]
  if sensor is None:
    return torch.zeros(env.num_envs, device=env.device)
  found = sensor.data.found
  assert found is not None
  count = (found > 0).float().flatten(start_dim=1).sum(dim=1)
  return torch.clamp(count / max(normalize_count, 1e-6), min=0.0, max=1.0)


def _body_quat_w(asset: Entity) -> torch.Tensor:
  quat = getattr(asset.data, "body_link_quat_w", None)
  if quat is not None:
    return quat
  quat = getattr(asset.data, "body_quat_w", None)
  if quat is not None:
    return quat
  raise AttributeError("robot data must expose body_link_quat_w or body_quat_w")


def _body_lin_vel_w(asset: Entity) -> torch.Tensor:
  vel = getattr(asset.data, "body_link_lin_vel_w", None)
  if vel is not None:
    return vel
  vel = getattr(asset.data, "body_lin_vel_w", None)
  if vel is not None:
    return vel
  raise AttributeError("robot data must expose body_link_lin_vel_w or body_lin_vel_w")


def _foot_contact_weights(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str | None,
  *,
  num_feet: int,
) -> torch.Tensor:
  if feet_sensor_name is None:
    return torch.ones(env.num_envs, num_feet, device=env.device)
  sensor = env.scene.get(feet_sensor_name) if isinstance(env.scene, dict) else env.scene[feet_sensor_name]
  if sensor is None:
    return torch.ones(env.num_envs, num_feet, device=env.device)
  found = sensor.data.found
  assert found is not None
  contact = (found > 0).float().flatten(start_dim=1)
  if contact.shape[1] == num_feet:
    return contact
  if contact.shape[1] >= 2 * num_feet and contact.shape[1] % num_feet == 0:
    return torch.clamp(contact.reshape(contact.shape[0], num_feet, -1).amax(dim=2), max=1.0)
  if contact.shape[1] >= 2 and num_feet == 2:
    return contact[:, :2]
  if contact.shape[1] == 1:
    return contact.expand(-1, num_feet)
  return torch.ones(env.num_envs, num_feet, device=env.device)


def _standing_gate(
  env: ManagerBasedRlEnv,
  *,
  min_height: float,
  min_alignment: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  height_gate = torch.clamp((torso_height - min_height) / 0.15, min=0.0, max=1.0)
  alignment_gate = torch.clamp(
    (_upright_alignment(asset.data.projected_gravity_b) - min_alignment)
    / max(1.0 - min_alignment, 1e-6),
    min=0.0,
    max=1.0,
  )
  return height_gate * alignment_gate


def _foot_flatness_from_quat(foot_quat_w: torch.Tensor) -> torch.Tensor:
  gravity_w = torch.zeros_like(foot_quat_w[..., :3])
  gravity_w[..., 2] = -1.0
  gravity_b = quat_apply_inverse(foot_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3))
  gravity_b = gravity_b.reshape(*foot_quat_w.shape[:-1], 3)
  # BFM-Zero penalizes lateral gravity components in the foot frame.  A flat
  # sole has local gravity almost purely along -Z, while toe-tip/rolled contacts
  # expose large x/y components.
  tilt = torch.linalg.norm(gravity_b[..., :2], dim=-1)
  return torch.clamp(1.0 - tilt, min=0.0, max=1.0)


def _heading_alignment_from_quat(root_quat_w: torch.Tensor, foot_quat_w: torch.Tensor) -> torch.Tensor:
  forward = torch.zeros_like(foot_quat_w[..., :3])
  forward[..., 0] = 1.0
  root_forward = quat_apply(root_quat_w, forward[:, 0, :] if forward.ndim == 3 else forward)
  foot_forward = quat_apply(foot_quat_w.reshape(-1, 4), forward.reshape(-1, 3)).reshape_as(forward)
  root_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
  foot_heading = torch.atan2(foot_forward[..., 1], foot_forward[..., 0])
  diff = torch.abs(wrap_to_pi(foot_heading - root_heading[:, None]))
  return torch.exp(-torch.square(diff / 0.6))


def _host_orientation_term(
  projected_gravity_b: torch.Tensor,
  *,
  orientation_threshold: float = 0.99,
  margin: float = 0.05,
) -> torch.Tensor:
  alignment = _upright_alignment(projected_gravity_b)
  miss = torch.clamp(orientation_threshold - alignment, min=0.0)
  return torch.exp(-torch.square(miss / max(margin, 1e-6)))


def _host_height_term(
  torso_height: torch.Tensor,
  *,
  target_base_height_phase1: float = 0.45,
  target_base_height_phase3: float = 0.65,
) -> torch.Tensor:
  return torch.clamp(
    (torso_height - target_base_height_phase1)
    / max(target_base_height_phase3 - target_base_height_phase1, 1e-6),
    min=0.0,
    max=1.0,
  )


def host_getup_task_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  orientation_threshold: float = 0.99,
  orientation_margin: float = 0.05,
  target_base_height_phase1: float = 0.45,
  target_base_height_phase3: float = 0.65,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 1.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Strict HoST end-state reward (kept multiplicative for anti-reward-hack).

  Paid only when the policy is simultaneously upright, tall, on its feet and
  not body-supported.  This is the *terminal* part of the get-up curriculum.
  Use ``host_getup_lift_progress_reward`` for the dense progress signal that
  shapes exploration from the fallen state.
  """

  asset: Entity = env.scene[asset_cfg.name]
  orientation = _host_orientation_term(
    asset.data.projected_gravity_b,
    orientation_threshold=orientation_threshold,
    margin=orientation_margin,
  )
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  height = _host_height_term(
    torso_height,
    target_base_height_phase1=target_base_height_phase1,
    target_base_height_phase3=target_base_height_phase3,
  )
  feet_contact = _contact_count(env, feet_sensor_name)
  body_support = _contact_count(env, body_sensor_name)
  feet_gate = torch.clamp(feet_contact / max(min_feet_contact_count, 1e-6), min=0.0, max=1.0)
  body_gate = 1.0 - torch.clamp(body_support / max(max_body_support_count, 1e-6), min=0.0, max=1.0)
  return orientation * height * feet_gate * body_gate


def host_getup_lift_progress_reward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.12,
  target_height: float = 0.55,
  orientation_floor: float = -1.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Dense early-stage progress reward to break the zero-gradient trap.

  ``host_getup_task_reward`` evaluates to exactly zero in the fallen start
  state (orientation gate ≈ 0, body gate = 0, feet gate = 0), so PPO had no
  exploration signal toward 'lift the torso'.  This term provides a soft,
  always-non-negative gradient that grows with torso height progress from
  the supine z (~0.1 m) up to the standing phase target (~0.55 m), weighted
  by upright alignment progress from ``orientation_floor`` upward so it
  doesn't reward purely passive falls.
  """
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  height_progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  alignment_progress = torch.clamp(
    (alignment - orientation_floor) / max(1.0 - orientation_floor, 1e-6),
    min=0.0,
    max=1.0,
  )
  # Height alone was enough for failed policies to learn a high-but-sideways
  # posture that never became a get-up.  Keep the dense height gradient, but
  # gate it by upright progress so torso lift only becomes highly valuable when
  # it is also rotating toward the success orientation.
  return height_progress * (0.25 + 0.75 * alignment_progress)


def host_upright_progress_reward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.18,
  target_height: float = 0.55,
  alignment_floor: float = -0.25,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Dense upright-rotation progress that is still tied to get-up height.

  Failed no-assist policies can lift the torso to ~0.50m while keeping the
  torso far from upright.  ``host_getup_lift_progress_reward`` intentionally
  gates height by orientation; this companion term provides the opposite
  gradient: rotate toward upright, but only with at least some lift progress so
  it does not reward idle rolling on the floor.
  """

  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  height_progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  alignment_progress = torch.clamp(
    (alignment - alignment_floor) / max(1.0 - alignment_floor, 1e-6),
    min=0.0,
    max=1.0,
  )
  return alignment_progress * (0.25 + 0.75 * height_progress)


def host_support_relief_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  min_height: float = 0.18,
  target_height: float = 0.55,
  max_body_support_count: float = 8.0,
  alignment_floor: float = -0.25,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Softly reward shifting support away from the torso/body stack.

  The strict HoST task term only pays when body support is nearly gone.  In the
  observed no-assist failures the policy reaches two foot contacts but remains
  heavily body-supported, so there is no usable gradient for progressively
  unloading the torso.  This term keeps the gradient soft and gated by lift,
  upright progress, and foot support; the strict completion reward remains the
  actual success criterion.
  """

  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  height_progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  alignment_progress = torch.clamp(
    (alignment - alignment_floor) / max(1.0 - alignment_floor, 1e-6),
    min=0.0,
    max=1.0,
  )

  feet_count = _contact_count(env, feet_sensor_name)
  feet_gate = 0.25 + 0.75 * torch.clamp(feet_count / 2.0, min=0.0, max=1.0)
  body_count = _contact_count(env, body_sensor_name)
  body_relief = 1.0 - torch.clamp(
    body_count / max(max_body_support_count, 1e-6),
    min=0.0,
    max=1.0,
  )
  return (
    body_relief
    * feet_gate
    * (0.25 + 0.75 * height_progress)
    * (0.25 + 0.75 * alignment_progress)
  )


def host_action_smoothness_penalty(
  env: ManagerBasedRlEnv,
  action_rate_weight: float = 1.0,
  smoothness_weight: float = 1.0,
  max_penalty: float = 50.0,
) -> torch.Tensor:
  """Bounded HoST action smoothness penalty.

  Without an upper bound, second-order smoothness over 23 dof at action-clip 5
  can reach ~9200 per step.  At weight -0.01 and scale_by_dt this still wipes
  out the host_task_reward, so clip per-step before the manager scales by dt.
  """

  action_manager = getattr(env, "action_manager", None)
  if action_manager is None:
    return torch.zeros(env.num_envs, device=env.device)
  action = getattr(env, "_host_getup_effective_action", None)
  prev_action = getattr(env, "_host_getup_prev_effective_action", None)
  prev_prev_action = getattr(env, "_host_getup_prev_prev_effective_action", None)
  if action is None or prev_action is None or prev_prev_action is None:
    action = action_manager.action
    prev_action = action_manager.prev_action
    prev_prev_action = action_manager.prev_prev_action
  rate = torch.sum(torch.square(action - prev_action), dim=1)
  smoothness = torch.sum(torch.square(action - 2.0 * prev_action + prev_prev_action), dim=1)
  penalty = action_rate_weight * rate + smoothness_weight * smoothness
  return _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)


def host_joint_tracking_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  target = getattr(env, "_host_getup_joint_position_target", None)
  target_ids = getattr(env, "_host_getup_joint_target_ids", None)
  if target is None or target_ids is None:
    return torch.zeros(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  current = asset.data.joint_pos[:, target_ids]
  return torch.sum(torch.square(target - current), dim=1)


def _select_existing_joint_ids(asset: Entity, joint_names: tuple[str, ...]) -> torch.Tensor:
  joint_name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
  return torch.tensor(
    [joint_name_to_index[name] for name in joint_names if name in joint_name_to_index],
    device=asset.data.joint_pos.device,
    dtype=torch.long,
  )


def host_style_pose_reward(
  env: ManagerBasedRlEnv,
  joint_names: tuple[str, ...],
  target_joint_angles: dict[str, float],
  std: float = 0.75,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
  min_height: float | None = None,
  min_alignment: float = 0.75,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = _select_existing_joint_ids(asset, joint_names)
  if joint_ids.numel() == 0:
    return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
  targets = torch.tensor(
    [target_joint_angles[asset.joint_names[int(joint_id)]] for joint_id in joint_ids],
    dtype=asset.data.joint_pos.dtype,
    device=asset.data.joint_pos.device,
  )
  error = asset.data.joint_pos[:, joint_ids] - targets.unsqueeze(0)
  reward = torch.exp(-torch.mean(torch.square(error), dim=1) / max(std**2, 1e-6))
  if min_height is None:
    return reward
  return reward * _standing_gate(
    env,
    min_height=min_height,
    min_alignment=min_alignment,
    asset_cfg=torso_asset_cfg,
  )


def host_feet_support_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  max_body_support_count: float = 2.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  feet_count = _contact_count(env, feet_sensor_name)
  body_count = _contact_count(env, body_sensor_name)
  asset: Entity = env.scene[asset_cfg.name]
  alignment = torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)
  feet_support = torch.clamp(feet_count / 2.0, min=0.0, max=1.0)
  body_relief = 1.0 - torch.clamp(body_count / max(max_body_support_count, 1e-6), min=0.0, max=1.0)
  height = _host_height_term(_torso_height(env, asset_cfg=asset_cfg))
  return feet_support * (0.25 + 0.75 * body_relief) * (0.25 + 0.75 * alignment) * (0.25 + 0.75 * height)


def host_hand_support_progress_reward(
  env: ManagerBasedRlEnv,
  hand_sensor_name: str,
  min_height: float = 0.18,
  release_height: float = 0.55,
  final_upright_threshold: float = 0.9,
  normalize_count: float = 2.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Reward hand contact only during the useful push-up phase.

  The current GetUp policy learned to stand but tends to twist into toe-tip
  support because hands are just another body contact that is penalized after
  lift.  This term makes palm/wrist ground contact useful while the torso is
  rising, then shuts off before final standing so the policy still releases
  the hands.
  """

  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  hand_support = _contact_norm(env, hand_sensor_name, normalize_count)
  height_progress = torch.clamp(
    (torso_height - min_height) / max(release_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  height_band = (0.25 + 0.75 * height_progress) * (torso_height >= min_height).float()
  below_release = (torso_height < release_height).float()
  not_final_upright = (_upright_alignment(asset.data.projected_gravity_b) < final_upright_threshold).float()
  return hand_support * height_band * below_release * not_final_upright


def host_hand_contact_after_stand_penalty(
  env: ManagerBasedRlEnv,
  hand_sensor_name: str,
  activation_height: float = 0.55,
  upright_alignment_threshold: float = 0.9,
  normalize_count: float = 2.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Penalize palm/wrist ground contact after the robot is upright and tall."""

  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  upright = _upright_alignment(asset.data.projected_gravity_b) >= upright_alignment_threshold
  standing = (torso_height >= activation_height) & upright
  return _contact_norm(env, hand_sensor_name, normalize_count) * standing.float()


def host_hand_push_reward(
  env: ManagerBasedRlEnv,
  hand_sensor_name: str,
  min_height: float = 0.18,
  release_height: float = 0.55,
  vertical_velocity_scale: float = 0.5,
  normalize_count: float = 2.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Reward hand contact only when it helps lift the torso."""

  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  torso_vel_z = _body_lin_vel_w(asset)[:, asset_cfg.body_ids, 2].amax(dim=1)
  hand_support = _contact_norm(env, hand_sensor_name, normalize_count)
  height_progress = torch.clamp(
    (torso_height - min_height) / max(release_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  height_band = (torso_height >= min_height).float() * (torso_height < release_height).float()
  upward = torch.clamp(torso_vel_z / max(vertical_velocity_scale, 1e-6), min=0.0, max=1.0)
  return hand_support * height_band * (0.5 + 0.5 * height_progress) * upward


def host_foot_flat_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str | None = "feet_ground_contact",
  min_height: float = 0.45,
  min_alignment: float = 0.5,
  foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """BFM-inspired flat-foot reward using ankle/foot orientation in contact.

  This is a positive counterpart of BFM-Zero's feet-orientation penalty:
  contacted feet get high reward when the local gravity vector is close to the
  foot -Z axis, which discourages toe-tip and heavily rolled support.
  """

  asset: Entity = env.scene[foot_asset_cfg.name]
  foot_quat = _body_quat_w(asset)[:, foot_asset_cfg.body_ids, :]
  flatness = _foot_flatness_from_quat(foot_quat)
  contact = _foot_contact_weights(env, feet_sensor_name, num_feet=flatness.shape[1])
  contact_gate = torch.clamp(contact.sum(dim=1) / max(float(flatness.shape[1]), 1.0), min=0.0, max=1.0)
  contacted_flatness = torch.where(contact > 0.0, flatness, torch.ones_like(flatness))
  both_feet_flat = contacted_flatness.amin(dim=1)
  return both_feet_flat * contact_gate * _standing_gate(
    env,
    min_height=min_height,
    min_alignment=min_alignment,
    asset_cfg=torso_asset_cfg,
  )


def host_foot_orientation_penalty(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str | None = "feet_ground_contact",
  min_height: float = 0.50,
  min_alignment: float = 0.75,
  foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """BFM-style contacted-foot orientation penalty for toe-tip/rolled stance."""

  asset: Entity = env.scene[foot_asset_cfg.name]
  foot_quat = _body_quat_w(asset)[:, foot_asset_cfg.body_ids, :]
  flatness = _foot_flatness_from_quat(foot_quat)
  contact = _foot_contact_weights(env, feet_sensor_name, num_feet=flatness.shape[1])
  contact_sum = torch.clamp(contact.sum(dim=1), min=1.0)
  contacted_tilt = (1.0 - flatness) * contact
  penalty = contacted_tilt.sum(dim=1) / contact_sum
  return penalty * _standing_gate(
    env,
    min_height=min_height,
    min_alignment=min_alignment,
    asset_cfg=torso_asset_cfg,
  )


def host_foot_heading_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str | None = "feet_ground_contact",
  min_height: float = 0.45,
  min_alignment: float = 0.5,
  foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Reward feet pointing with the root heading in the final stand phase."""

  asset: Entity = env.scene[foot_asset_cfg.name]
  foot_quat = _body_quat_w(asset)[:, foot_asset_cfg.body_ids, :]
  heading = _heading_alignment_from_quat(asset.data.root_link_quat_w, foot_quat)
  contact = _foot_contact_weights(env, feet_sensor_name, num_feet=heading.shape[1])
  contact_gate = torch.clamp(contact.sum(dim=1) / max(float(heading.shape[1]), 1.0), min=0.0, max=1.0)
  contacted_heading = torch.where(contact > 0.0, heading, torch.ones_like(heading))
  all_feet_heading = contacted_heading.amin(dim=1)
  return all_feet_heading * contact_gate * _standing_gate(
    env,
    min_height=min_height,
    min_alignment=min_alignment,
    asset_cfg=torso_asset_cfg,
  )


def host_foot_contact_spread_reward(
  env: ManagerBasedRlEnv,
  foot_geom_sensor_name: str,
  feet_sensor_name: str = "feet_ground_contact",
  min_height: float = 0.35,
  target_contacts_per_foot: float = 3.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Reward multiple sole contact geoms instead of a single toe-tip contact."""

  sensor = env.scene[foot_geom_sensor_name]
  found = sensor.data.found
  assert found is not None
  contact = (found > 0).float().flatten(start_dim=1)
  if contact.shape[1] >= 14:
    foot_contacts = torch.stack([contact[:, :7].sum(dim=1), contact[:, 7:14].sum(dim=1)], dim=1)
  else:
    coarse = _foot_contact_weights(env, feet_sensor_name, num_feet=2)
    foot_contacts = coarse * target_contacts_per_foot
  spread = torch.clamp(foot_contacts / max(target_contacts_per_foot, 1e-6), min=0.0, max=1.0)
  both_feet_gate = torch.clamp(_contact_count(env, feet_sensor_name) / 2.0, min=0.0, max=1.0)
  height_gate = torch.clamp((_torso_height(env, asset_cfg=asset_cfg) - min_height) / 0.25, min=0.0, max=1.0)
  both_feet_spread = 0.5 * spread.mean(dim=1) + 0.5 * spread.amin(dim=1)
  return both_feet_spread * both_feet_gate * height_gate


def host_ankle_deviation_penalty(
  env: ManagerBasedRlEnv,
  joint_names: tuple[str, ...],
  target_joint_angles: dict[str, float],
  std: float = 0.35,
  min_height: float = 0.50,
  min_alignment: float = 0.75,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Penalize final-stance ankle pitch/roll deviations that create toe stands."""

  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = _select_existing_joint_ids(asset, joint_names)
  if joint_ids.numel() == 0:
    return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)
  targets = torch.tensor(
    [target_joint_angles[asset.joint_names[int(joint_id)]] for joint_id in joint_ids],
    dtype=asset.data.joint_pos.dtype,
    device=asset.data.joint_pos.device,
  )
  error = torch.mean(torch.abs(asset.data.joint_pos[:, joint_ids] - targets.unsqueeze(0)), dim=1)
  penalty = torch.clamp(error / max(std, 1e-6), min=0.0, max=1.0)
  return penalty * _standing_gate(
    env,
    min_height=min_height,
    min_alignment=min_alignment,
    asset_cfg=torso_asset_cfg,
  )


def host_natural_stand_pose_reward(
  env: ManagerBasedRlEnv,
  joint_names: tuple[str, ...],
  target_joint_angles: dict[str, float],
  std: float = 0.35,
  min_height: float = 0.55,
  min_alignment: float = 0.75,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Tight final-pose reward for untwisted legs and neutral ankles."""

  pose = host_style_pose_reward(
    env,
    joint_names=joint_names,
    target_joint_angles=target_joint_angles,
    std=std,
    asset_cfg=asset_cfg,
  )
  return pose * _standing_gate(
    env,
    min_height=min_height,
    min_alignment=min_alignment,
    asset_cfg=torso_asset_cfg,
  )


def host_target_standing_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  base_height_target: float = 0.75,
  target_base_height_phase3: float = 0.65,
  standing_gate_start_height: float | None = None,
  max_body_support_count: float = 8.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Soft standing-pose reward.

  The original exp(-20*|h-target|) collapses to zero for any miss beyond ~15 cm,
  giving no gradient early.  Use a gentler height kernel (std ~0.2 m) and a
  soft feet/body gate so partial progress still gets credit.
  """
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  feet_count = _contact_count(env, feet_sensor_name)
  body_count = _contact_count(env, body_sensor_name)
  gate_start = target_base_height_phase3 if standing_gate_start_height is None else standing_gate_start_height
  height_progress = torch.clamp(
    (torso_height - gate_start) / max(base_height_target - gate_start, 1e-6),
    min=0.0,
    max=1.0,
  )
  alignment = torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)
  feet_gate = 0.25 + 0.75 * torch.clamp(feet_count / 2.0, min=0.0, max=1.0)
  body_relief = 1.0 - torch.clamp(body_count / max(max_body_support_count, 1e-6), min=0.0, max=1.0)
  orientation = 0.25 + 0.75 * alignment
  body_gate = 0.25 + 0.75 * body_relief
  return orientation * height_progress * feet_gate * body_gate


def getup_posture_reward(
  env: ManagerBasedRlEnv,
  tilt_std: float = 0.35,
  torso_height_target: float = 0.62,
  torso_height_std: float = 0.2,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  upright_alignment = torch.clamp(_upright_alignment(projected_gravity_b), min=0.0, max=1.0)
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  tilt_term = torch.exp(-torch.square(tilt) / max(tilt_std**2, 1e-6))
  height_term = torch.exp(-torch.square(torso_height - torso_height_target) / max(torso_height_std**2, 1e-6))
  return tilt_term * height_term * upright_alignment


def getup_height_progress_reward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.18,
  target_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b
  upright_alignment = torch.clamp(_upright_alignment(projected_gravity_b), min=0.0, max=1.0)
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  return progress * upright_alignment


def getup_torso_lift_reward(
  env: ManagerBasedRlEnv,
  min_height: float = 0.12,
  target_height: float = 0.55,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  return torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )


def getup_facing_up_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)


class getup_phase_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._paid_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    thresholds: tuple[float, ...] = (0.22, 0.4, 0.55),
    bonuses: tuple[float, ...] = (1.0, 2.0, 3.0),
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    assert len(thresholds) == len(bonuses), "thresholds and bonuses must align"
    torso_height = _torso_height(env, asset_cfg=asset_cfg)
    reward = torch.zeros(env.num_envs, device=env.device)
    for stage_idx, (threshold, bonus) in enumerate(zip(thresholds, bonuses, strict=False), start=1):
      newly_crossed = (self._paid_stage < stage_idx) & (torso_height >= threshold)
      reward = reward + newly_crossed.float() * bonus
      self._paid_stage[newly_crossed] = stage_idx
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid_stage[env_ids] = 0


class getup_orientation_phase_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._paid_stage = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    thresholds: tuple[float, ...] = (0.1, 0.4, 0.7),
    bonuses: tuple[float, ...] = (1.0, 2.0, 3.0),
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    assert len(thresholds) == len(bonuses), "thresholds and bonuses must align"
    asset: Entity = env.scene[asset_cfg.name]
    alignment = torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)
    reward = torch.zeros(env.num_envs, device=env.device)
    for stage_idx, (threshold, bonus) in enumerate(zip(thresholds, bonuses, strict=False), start=1):
      newly_crossed = (self._paid_stage < stage_idx) & (alignment >= threshold)
      reward = reward + newly_crossed.float() * bonus
      self._paid_stage[newly_crossed] = stage_idx
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid_stage[env_ids] = 0


class reduced_support_bonus:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_support_count: float = 0.5,
    activation_height: float = 0.4,
    alignment_threshold: float = 0.3,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    support_count = (sensor_data.found > 0).float().sum(dim=1)
    asset: Entity = env.scene[asset_cfg.name]
    torso_height = _torso_height(env, asset_cfg=asset_cfg)
    facing_up = _upright_alignment(asset.data.projected_gravity_b) >= alignment_threshold
    eligible = (torso_height >= activation_height) & facing_up & (support_count <= max_support_count) & ~self._paid
    reward = eligible.float()
    self._paid[eligible] = True
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._paid[env_ids] = False


class getup_demo_pose_reward:
  def __init__(
    self,
    cfg,
    env: ManagerBasedRlEnv,
    demo_npz_path: str | None = None,
    joint_names: tuple[str, ...] | None = None,
    dt_per_demo_frame: float | None = None,
  ):
    params = getattr(cfg, "params", {}) or {}
    if demo_npz_path is None:
      demo_npz_path = params.get("demo_npz_path")
    if joint_names is None:
      joint_names = params.get("joint_names")
    if dt_per_demo_frame is None:
      dt_per_demo_frame = params.get("dt_per_demo_frame", 0.02)
    if demo_npz_path is None or joint_names is None:
      raise TypeError("getup_demo_pose_reward requires demo_npz_path and joint_names")
    self._elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    payload = np.load(Path(demo_npz_path).expanduser())
    joint_pos = torch.tensor(payload["joint_pos"], dtype=torch.float32, device=env.device)
    self._demo_joint_pos = joint_pos
    asset: Entity = env.scene[_JOINT_ASSET_CFG.name]
    joint_name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    self._joint_ids = torch.tensor(
      [joint_name_to_index[name] for name in joint_names if name in joint_name_to_index],
      dtype=torch.long,
      device=env.device,
    )
    step_dt = getattr(env, "step_dt", dt_per_demo_frame)
    self._frames_per_step = max(1, int(round(step_dt / max(dt_per_demo_frame, 1e-6))))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    demo_npz_path: str,
    joint_names: tuple[str, ...],
    dt_per_demo_frame: float = 0.02,
    std: float = 0.75,
  ) -> torch.Tensor:
    del demo_npz_path, joint_names, dt_per_demo_frame
    frame_idx = torch.clamp(self._elapsed_steps * self._frames_per_step, max=self._demo_joint_pos.shape[0] - 1)
    target = self._demo_joint_pos[frame_idx][:, self._joint_ids]
    asset: Entity = env.scene[_JOINT_ASSET_CFG.name]
    current = asset.data.joint_pos[:, self._joint_ids]
    reward = torch.exp(-torch.mean(torch.square(current - target), dim=1) / max(std**2, 1e-6))
    self._elapsed_steps += 1
    return reward

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._elapsed_steps[env_ids] = 0


def stand_still_after_getup(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  activation_height: float = 0.45,
  facing_up_threshold: float = 0.3,
  torso_asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  joint_asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[torso_asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=torso_asset_cfg)
  facing_up = _upright_alignment(asset.data.projected_gravity_b) >= facing_up_threshold
  active = (torso_height >= activation_height) & facing_up
  penalty = _base_stand_still(
    env,
    command_name=command_name,
    command_threshold=command_threshold,
    asset_cfg=joint_asset_cfg,
  )
  return penalty * active.float()


def action_rate_after_lift(
  env: ManagerBasedRlEnv,
  activation_height: float = 0.25,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  active = (torso_height >= activation_height).float()
  penalty = envs_mdp.action_rate_l2(env)
  return penalty * active


def bounded_action_rate_after_lift(
  env: ManagerBasedRlEnv,
  activation_height: float = 0.25,
  max_penalty: float = 250.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  active = (torso_height >= activation_height).float()
  penalty = envs_mdp.action_rate_l2(env)
  penalty = _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)
  return penalty * active


def track_linear_velocity_after_lift(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  activation_height: float = 0.45,
  alignment_threshold: float = 0.3,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  facing_up = _upright_alignment(asset.data.projected_gravity_b) >= alignment_threshold
  active = (torso_height >= activation_height) & facing_up
  reward = _base_track_linear_velocity(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
  return reward * active.float()


def track_angular_velocity_after_lift(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  activation_height: float = 0.45,
  alignment_threshold: float = 0.3,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  facing_up = _upright_alignment(asset.data.projected_gravity_b) >= alignment_threshold
  active = (torso_height >= activation_height) & facing_up
  reward = _base_track_angular_velocity(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
  return reward * active.float()


def support_contact_diversity_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_count: float = 2.0,
  active_below_height: float | None = None,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  contact_count = (sensor_data.found > 0).float().sum(dim=1)
  reward = torch.exp(-torch.square(contact_count - target_count))
  if active_below_height is not None:
    torso_height = _torso_height(env, asset_cfg=asset_cfg)
    reward = reward * (torso_height < active_below_height).float()
  return reward


def support_body_contact_penalty_after_lift(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  hand_sensor_name: str | None = None,
  activation_height: float = 0.35,
  hand_release_height: float = 0.55,
  normalize_count: float = 2.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  contact_count = (sensor_data.found > 0).float().flatten(start_dim=1).sum(dim=1)
  if hand_sensor_name is not None:
    hand_contact = _contact_norm(env, hand_sensor_name, normalize_count=normalize_count)
    hand_count = hand_contact * normalize_count
    torso_height_for_release = _torso_height(env, asset_cfg=asset_cfg)
    hand_allowed = (torso_height_for_release < hand_release_height).float()
    contact_count = torch.clamp(contact_count - hand_count * hand_allowed, min=0.0)
  torso_height = _torso_height(env, asset_cfg=asset_cfg)
  active = (torso_height >= activation_height).float()
  normalized = torch.clamp(contact_count / max(normalize_count, 1e-6), min=0.0, max=1.0)
  return normalized * active


def getup_feet_support_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  max_body_support_count: float = 4.0,
  alignment_floor: float = -0.1,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  feet_sensor = env.scene[feet_sensor_name]
  feet_sensor_data = feet_sensor.data
  assert feet_sensor_data.found is not None
  feet_count = (feet_sensor_data.found > 0).float().sum(dim=1)
  feet_support = torch.clamp(feet_count / 2.0, min=0.0, max=1.0)

  body_sensor = env.scene[body_sensor_name]
  body_sensor_data = body_sensor.data
  assert body_sensor_data.found is not None
  body_count = (body_sensor_data.found > 0).float().sum(dim=1)
  body_relief = 1.0 - torch.clamp(body_count / max(max_body_support_count, 1e-6), min=0.0, max=1.0)

  asset: Entity = env.scene[asset_cfg.name]
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  alignment_progress = torch.clamp(
    (alignment - alignment_floor) / max(1.0 - alignment_floor, 1e-6),
    min=0.0,
    max=1.0,
  )
  return feet_support * (0.25 + 0.75 * body_relief) * (0.25 + 0.75 * alignment_progress)


def getup_standing_joint_pose_reward(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  joint_names: tuple[str, ...],
  std: float = 0.75,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 2.0,
  alignment_threshold: float = 0.0,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  joint_name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
  selected_joint_ids = [joint_name_to_index[name] for name in joint_names if name in joint_name_to_index]
  if not selected_joint_ids:
    return torch.zeros(asset.data.joint_pos.shape[0], device=asset.data.joint_pos.device)

  selected_joint_ids_t = torch.tensor(selected_joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
  joint_error = asset.data.joint_pos[:, selected_joint_ids_t] - asset.data.default_joint_pos[:, selected_joint_ids_t]
  pose_reward = torch.exp(-torch.mean(torch.square(joint_error), dim=1) / max(std**2, 1e-6))

  feet_sensor = env.scene[feet_sensor_name]
  feet_found = feet_sensor.data.found
  assert feet_found is not None
  feet_contact_count = (feet_found > 0).float().sum(dim=1)

  body_sensor = env.scene[body_sensor_name]
  body_found = body_sensor.data.found
  assert body_found is not None
  body_support_count = (body_found > 0).float().sum(dim=1)

  alignment = _upright_alignment(asset.data.projected_gravity_b)
  active = (
    (feet_contact_count >= min_feet_contact_count)
    & (body_support_count <= max_body_support_count)
    & (alignment >= alignment_threshold)
  )
  return pose_reward * active.float()


def joint_pos_limits_after_support(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  body_sensor_name: str,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 2.0,
  alignment_threshold: float = 0.0,
  max_penalty: float = 10.0,
  asset_cfg: SceneEntityCfg = _JOINT_ASSET_CFG,
) -> torch.Tensor:
  feet_sensor = env.scene[feet_sensor_name]
  feet_found = feet_sensor.data.found
  assert feet_found is not None
  feet_contact_count = (feet_found > 0).float().sum(dim=1)

  body_sensor = env.scene[body_sensor_name]
  body_found = body_sensor.data.found
  assert body_found is not None
  body_support_count = (body_found > 0).float().sum(dim=1)

  asset: Entity = env.scene[asset_cfg.name]
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  active = (
    (feet_contact_count >= min_feet_contact_count)
    & (body_support_count <= max_body_support_count)
    & (alignment >= alignment_threshold)
  )
  penalty = base_envs_mdp.joint_pos_limits(env, asset_cfg=asset_cfg)
  penalty = _bounded_nonnegative_penalty(penalty, max_penalty=max_penalty)
  return penalty * active.float()


def self_collision_cost_after_support(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  feet_sensor_name: str,
  body_sensor_name: str,
  force_threshold: float = 10.0,
  min_feet_contact_count: float = 1.0,
  max_body_support_count: float = 2.0,
  alignment_threshold: float = 0.0,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  feet_sensor = env.scene[feet_sensor_name]
  feet_found = feet_sensor.data.found
  assert feet_found is not None
  feet_contact_count = (feet_found > 0).float().sum(dim=1)

  body_sensor = env.scene[body_sensor_name]
  body_found = body_sensor.data.found
  assert body_found is not None
  body_support_count = (body_found > 0).float().sum(dim=1)

  asset: Entity = env.scene[asset_cfg.name]
  alignment = _upright_alignment(asset.data.projected_gravity_b)
  active = (
    (feet_contact_count >= min_feet_contact_count)
    & (body_support_count <= max_body_support_count)
    & (alignment >= alignment_threshold)
  )
  return _base_self_collision_cost(env, sensor_name=sensor_name, force_threshold=force_threshold) * active.float()


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
