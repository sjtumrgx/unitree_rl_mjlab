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
_FOOT_ASSET_CFG = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))


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


def _scene_get(env: ManagerBasedRlEnv, name: str):
  if name == "env_origins" and hasattr(env.scene, "env_origins"):
    return env.scene.env_origins
  if isinstance(getattr(env, "scene", None), dict):
    return env.scene.get(name)
  try:
    return env.scene[name]
  except Exception:
    return None


def _body_quat_w(asset: Entity) -> torch.Tensor | None:
  quat = getattr(asset.data, "body_link_quat_w", None)
  if quat is not None:
    return quat
  return getattr(asset.data, "body_quat_w", None)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  q = q.float()
  v = v.float()
  q_w = q[..., :1]
  q_vec = q[..., 1:]
  return v * (2.0 * q_w * q_w - 1.0) + 2.0 * q_w * torch.cross(q_vec, v, dim=-1) + 2.0 * q_vec * (
    q_vec * v
  ).sum(dim=-1, keepdim=True)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  inv = q.clone()
  inv[..., 1:] = -inv[..., 1:]
  return _quat_apply(inv, v)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _contact_count_tensor(
  env: ManagerBasedRlEnv,
  sensor_name: str | None,
  *,
  num_envs: int,
  device: torch.device | str,
) -> torch.Tensor | None:
  if sensor_name is None:
    return None
  sensor = _scene_get(env, sensor_name)
  found = getattr(getattr(sensor, "data", None), "found", None)
  if found is None:
    return None
  return (found > 0).float().flatten(start_dim=1).sum(dim=1).to(device=device)


def _foot_contact_weights(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str | None,
  *,
  num_feet: int,
  num_envs: int,
  device: torch.device | str,
) -> torch.Tensor:
  if feet_sensor_name is None:
    return torch.ones(num_envs, num_feet, device=device)
  sensor = _scene_get(env, feet_sensor_name)
  found = getattr(getattr(sensor, "data", None), "found", None)
  if found is None:
    return torch.ones(num_envs, num_feet, device=device)
  contact = (found > 0).float().flatten(start_dim=1).to(device=device)
  if contact.shape[1] == num_feet:
    return contact
  if contact.shape[1] >= 2 * num_feet and contact.shape[1] % num_feet == 0:
    return torch.clamp(contact.reshape(contact.shape[0], num_feet, -1).amax(dim=2), max=1.0)
  if contact.shape[1] >= 2 and num_feet == 2:
    return contact[:, :2]
  if contact.shape[1] == 1:
    return contact.expand(-1, num_feet)
  return torch.ones(num_envs, num_feet, device=device)


def _foot_flatness_from_quat(foot_quat_w: torch.Tensor) -> torch.Tensor:
  gravity_w = torch.zeros_like(foot_quat_w[..., :3])
  gravity_w[..., 2] = -1.0
  gravity_b = _quat_apply_inverse(foot_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3))
  gravity_b = gravity_b.reshape(*foot_quat_w.shape[:-1], 3)
  tilt = torch.linalg.norm(gravity_b[..., :2], dim=-1)
  return torch.clamp(1.0 - tilt, min=0.0, max=1.0)


def _heading_alignment_from_quat(root_quat_w: torch.Tensor, foot_quat_w: torch.Tensor) -> torch.Tensor:
  forward = torch.zeros_like(foot_quat_w[..., :3])
  forward[..., 0] = 1.0
  root_forward = _quat_apply(root_quat_w, forward[:, 0, :] if forward.ndim == 3 else forward)
  foot_forward = _quat_apply(foot_quat_w.reshape(-1, 4), forward.reshape(-1, 3)).reshape_as(forward)
  root_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
  foot_heading = torch.atan2(foot_forward[..., 1], foot_forward[..., 0])
  diff = torch.abs(_wrap_to_pi(foot_heading - root_heading[:, None]))
  return torch.exp(-torch.square(diff / 0.6))


def _foot_geom_spread(
  env: ManagerBasedRlEnv,
  sensor_name: str | None,
  *,
  num_envs: int,
  device: torch.device | str,
  target_contacts_per_foot: float = 3.0,
) -> torch.Tensor | None:
  if sensor_name is None:
    return None
  sensor = _scene_get(env, sensor_name)
  found = getattr(getattr(sensor, "data", None), "found", None)
  if found is None:
    return None
  contact = (found > 0).float().flatten(start_dim=1).to(device=device)
  if contact.shape[1] < 14:
    return None
  left = torch.clamp(contact[:, :7].sum(dim=1) / max(target_contacts_per_foot, 1e-6), min=0.0, max=1.0)
  right = torch.clamp(contact[:, 7:14].sum(dim=1) / max(target_contacts_per_foot, 1e-6), min=0.0, max=1.0)
  return torch.stack([left, right], dim=1)


def stable_getup_success_mask(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  *,
  tilt_threshold: float = 0.3,
  torso_height_threshold: float = 0.55,
  feet_sensor_name: str | None = None,
  body_sensor_name: str | None = None,
  hand_sensor_name: str | None = None,
  foot_geom_sensor_name: str | None = None,
  min_feet_contact_count: float = 0.0,
  max_body_support_count: float | None = None,
  max_hand_contact_count: float | None = None,
  min_foot_flatness: float | None = None,
  min_foot_heading_alignment: float | None = None,
  min_foot_geom_contact_spread: float | None = None,
  foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  """Shared strict GetUp success definition for rewards, metrics and assist decay.

  The legacy success signal only checked torso height/uprightness, which let
  policies receive completion credit while balancing on toe tips, hands, or
  other body contacts.  Optional contact/posture gates make the configured
  training and diagnostics agree on "stable standing".
  """

  asset: Entity = env.scene[asset_cfg.name]
  torso_height = _relative_body_height(env, asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]).amax(dim=1)
  projected_gravity_b = asset.data.projected_gravity_b
  tilt = torch.linalg.norm(projected_gravity_b[:, :2], dim=1)
  success = (tilt <= tilt_threshold) & _is_facing_up(projected_gravity_b, tilt_threshold=tilt_threshold) & (
    torso_height >= torso_height_threshold
  )
  device = torso_height.device
  num_envs = int(torso_height.shape[0])

  feet_count = _contact_count_tensor(env, feet_sensor_name, num_envs=num_envs, device=device)
  if feet_count is not None:
    success &= feet_count >= float(min_feet_contact_count)

  body_count = _contact_count_tensor(env, body_sensor_name, num_envs=num_envs, device=device)
  if body_count is not None and max_body_support_count is not None:
    success &= body_count <= float(max_body_support_count)

  hand_count = _contact_count_tensor(env, hand_sensor_name, num_envs=num_envs, device=device)
  if hand_count is not None and max_hand_contact_count is not None:
    success &= hand_count <= float(max_hand_contact_count)

  foot_asset: Entity = env.scene[foot_asset_cfg.name]
  foot_quat = _body_quat_w(foot_asset)
  if foot_quat is not None and (
    min_foot_flatness is not None or min_foot_heading_alignment is not None
  ):
    foot_quat = foot_quat[:, foot_asset_cfg.body_ids, :]
    contact = _foot_contact_weights(
      env,
      feet_sensor_name,
      num_feet=foot_quat.shape[1],
      num_envs=num_envs,
      device=device,
    )
    if min_foot_flatness is not None:
      flatness = _foot_flatness_from_quat(foot_quat)
      contacted_flatness = torch.where(contact > 0.0, flatness, torch.zeros_like(flatness))
      success &= contacted_flatness.amin(dim=1) >= float(min_foot_flatness)
    if min_foot_heading_alignment is not None:
      heading = _heading_alignment_from_quat(foot_asset.data.root_link_quat_w, foot_quat)
      contacted_heading = torch.where(contact > 0.0, heading, torch.zeros_like(heading))
      success &= contacted_heading.amin(dim=1) >= float(min_foot_heading_alignment)

  spread = _foot_geom_spread(env, foot_geom_sensor_name, num_envs=num_envs, device=device)
  if spread is not None and min_foot_geom_contact_spread is not None:
    success &= spread.amin(dim=1) >= float(min_foot_geom_contact_spread)

  if env_ids is None:
    return success
  return success[env_ids.to(device=device, dtype=torch.long)]




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
  feet_sensor_name: str | None = None,
  body_sensor_name: str | None = None,
  hand_sensor_name: str | None = None,
  foot_geom_sensor_name: str | None = None,
  min_feet_contact_count: float = 0.0,
  max_body_support_count: float | None = None,
  max_hand_contact_count: float | None = None,
  min_foot_flatness: float | None = None,
  min_foot_heading_alignment: float | None = None,
  min_foot_geom_contact_spread: float | None = None,
  foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
  asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
) -> torch.Tensor:
  return stable_getup_success_mask(
    env,
    tilt_threshold=tilt_threshold,
    torso_height_threshold=torso_height_threshold,
    feet_sensor_name=feet_sensor_name,
    body_sensor_name=body_sensor_name,
    hand_sensor_name=hand_sensor_name,
    foot_geom_sensor_name=foot_geom_sensor_name,
    min_feet_contact_count=min_feet_contact_count,
    max_body_support_count=max_body_support_count,
    max_hand_contact_count=max_hand_contact_count,
    min_foot_flatness=min_foot_flatness,
    min_foot_heading_alignment=min_foot_heading_alignment,
    min_foot_geom_contact_spread=min_foot_geom_contact_spread,
    foot_asset_cfg=foot_asset_cfg,
    asset_cfg=asset_cfg,
  ).float()


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
    feet_sensor_name: str | None = None,
    body_sensor_name: str | None = None,
    hand_sensor_name: str | None = None,
    foot_geom_sensor_name: str | None = None,
    min_feet_contact_count: float = 0.0,
    max_body_support_count: float | None = None,
    max_hand_contact_count: float | None = None,
    min_foot_flatness: float | None = None,
    min_foot_heading_alignment: float | None = None,
    min_foot_geom_contact_spread: float | None = None,
    foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    success = getup_upright(
      env,
      tilt_threshold=tilt_threshold,
      torso_height_threshold=torso_height_threshold,
      feet_sensor_name=feet_sensor_name,
      body_sensor_name=body_sensor_name,
      hand_sensor_name=hand_sensor_name,
      foot_geom_sensor_name=foot_geom_sensor_name,
      min_feet_contact_count=min_feet_contact_count,
      max_body_support_count=max_body_support_count,
      max_hand_contact_count=max_hand_contact_count,
      min_foot_flatness=min_foot_flatness,
      min_foot_heading_alignment=min_foot_heading_alignment,
      min_foot_geom_contact_spread=min_foot_geom_contact_spread,
      foot_asset_cfg=foot_asset_cfg,
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
    feet_sensor_name: str | None = None,
    body_sensor_name: str | None = None,
    hand_sensor_name: str | None = None,
    foot_geom_sensor_name: str | None = None,
    min_feet_contact_count: float = 0.0,
    max_body_support_count: float | None = None,
    max_hand_contact_count: float | None = None,
    min_foot_flatness: float | None = None,
    min_foot_heading_alignment: float | None = None,
    min_foot_geom_contact_spread: float | None = None,
    foot_asset_cfg: SceneEntityCfg = _FOOT_ASSET_CFG,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    success = getup_upright(
      env,
      tilt_threshold=tilt_threshold,
      torso_height_threshold=torso_height_threshold,
      feet_sensor_name=feet_sensor_name,
      body_sensor_name=body_sensor_name,
      hand_sensor_name=hand_sensor_name,
      foot_geom_sensor_name=foot_geom_sensor_name,
      min_feet_contact_count=min_feet_contact_count,
      max_body_support_count=max_body_support_count,
      max_hand_contact_count=max_hand_contact_count,
      min_foot_flatness=min_foot_flatness,
      min_foot_heading_alignment=min_foot_heading_alignment,
      min_foot_geom_contact_spread=min_foot_geom_contact_spread,
      foot_asset_cfg=foot_asset_cfg,
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
