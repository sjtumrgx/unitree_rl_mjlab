from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

DISTURBANCE_NONE = 0
DISTURBANCE_PUSH = 1
DISTURBANCE_IMPULSE = 2
DISTURBANCE_NEAR_FAILURE_RESET = 3


def get_antifall_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  state = getattr(env, "_anti_fall_state", None)
  if state is not None and state["last_disturbance_step"].shape[0] == env.num_envs:
    return state

  state = {
    "last_disturbance_step": torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    ),
    "last_disturbance_mag": torch.zeros(env.num_envs, device=env.device),
    "disturbance_kind": torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    ),
    "disturbance_active": torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    ),
    "disturbance_count": torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    ),
  }
  setattr(env, "_anti_fall_state", state)
  return state


def disturbance_age_s(env: ManagerBasedRlEnv) -> torch.Tensor:
  state = get_antifall_state(env)
  step = int(getattr(env, "common_step_counter", 0))
  age_steps = torch.clamp(step - state["last_disturbance_step"], min=0).float()
  age_s = age_steps * env.step_dt
  return torch.where(
    state["last_disturbance_step"] >= 0,
    age_s,
    torch.full_like(age_s, float("inf")),
  )


def disturbance_window_mask(
  env: ManagerBasedRlEnv,
  window_s: float,
) -> torch.Tensor:
  state = get_antifall_state(env)
  age_s = disturbance_age_s(env)
  return (state["last_disturbance_step"] >= 0) & (age_s <= window_s)


def disturbance_age_fraction(
  env: ManagerBasedRlEnv,
  window_s: float,
) -> torch.Tensor:
  age_s = disturbance_age_s(env)
  frac = 1.0 - torch.clamp(age_s / max(window_s, env.step_dt), 0.0, 1.0)
  return torch.where(torch.isfinite(age_s), frac, torch.zeros_like(frac))


def reset_antifall_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None = None,
) -> None:
  if env_ids is None:
    env_ids = slice(None)
  state = get_antifall_state(env)
  state["last_disturbance_step"][env_ids] = -1
  state["last_disturbance_mag"][env_ids] = 0.0
  state["disturbance_kind"][env_ids] = DISTURBANCE_NONE
  state["disturbance_active"][env_ids] = False
  state["disturbance_count"][env_ids] = 0


def _resolve_env_ids(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    start = 0 if env_ids.start is None else env_ids.start
    stop = env.num_envs if env_ids.stop is None else env_ids.stop
    step = 1 if env_ids.step is None else env_ids.step
    return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
  return env_ids.to(device=env.device, dtype=torch.long)


def _mark_disturbance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  magnitude: torch.Tensor,
  kind: int,
  *,
  active: bool,
) -> None:
  ids = _resolve_env_ids(env, env_ids)
  if len(ids) == 0:
    return
  state = get_antifall_state(env)
  state["last_disturbance_step"][ids] = int(getattr(env, "common_step_counter", 0))
  state["last_disturbance_mag"][ids] = magnitude
  state["disturbance_kind"][ids] = kind
  state["disturbance_active"][ids] = active
  state["disturbance_count"][ids] += 1


def push_by_setting_velocity_with_history(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  recovery_window_s: float | None = None,
  active: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  del recovery_window_s
  if env_ids.numel() == 0:
    return
  asset = env.scene[asset_cfg.name]
  before = asset.data.root_link_vel_w[env_ids].clone()
  envs_mdp.push_by_setting_velocity(
    env,
    env_ids,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )
  after = asset.data.root_link_vel_w[env_ids]
  delta = after - before
  magnitude = torch.linalg.norm(delta[:, :3], dim=1)
  magnitude += 0.25 * torch.linalg.norm(delta[:, 3:], dim=1)
  _mark_disturbance(
    env,
    env_ids,
    magnitude=magnitude,
    kind=DISTURBANCE_PUSH,
    active=active,
  )


def scheduled_hard_reset_prob(
  hard_reset_prob: float,
  hard_reset_prob_schedule: tuple[dict[str, float], ...] | list[dict[str, float]] | None = None,
  *,
  common_step_counter: int,
) -> float:
  """Return hard-reset probability after applying an optional step schedule."""

  prob = float(hard_reset_prob)
  if hard_reset_prob_schedule is None:
    return prob
  for stage in hard_reset_prob_schedule:
    if int(common_step_counter) >= int(stage["step"]):
      prob = float(stage["prob"])
  return prob


def reset_root_state_mixed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  nominal_pose_range: dict[str, tuple[float, float]],
  nominal_velocity_range: dict[str, tuple[float, float]] | None = None,
  hard_pose_range: dict[str, tuple[float, float]] | None = None,
  hard_velocity_range: dict[str, tuple[float, float]] | None = None,
  hard_reset_prob: float = 0.0,
  hard_reset_prob_schedule: tuple[dict[str, float], ...] | list[dict[str, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  ids = _resolve_env_ids(env, env_ids)
  if len(ids) == 0:
    return

  reset_antifall_state(env, ids)

  hard_reset_prob = scheduled_hard_reset_prob(
    hard_reset_prob,
    hard_reset_prob_schedule,
    common_step_counter=int(getattr(env, "common_step_counter", 0)),
  )

  if hard_pose_range is None or hard_reset_prob <= 0.0:
    envs_mdp.reset_root_state_uniform(
      env,
      ids,
      pose_range=nominal_pose_range,
      velocity_range=nominal_velocity_range,
      asset_cfg=asset_cfg,
    )
    return

  hard_mask = torch.rand(len(ids), device=env.device) < hard_reset_prob
  hard_ids = ids[hard_mask]
  nominal_ids = ids[~hard_mask]

  if len(nominal_ids) > 0:
    envs_mdp.reset_root_state_uniform(
      env,
      nominal_ids,
      pose_range=nominal_pose_range,
      velocity_range=nominal_velocity_range,
      asset_cfg=asset_cfg,
    )

  if len(hard_ids) > 0:
    envs_mdp.reset_root_state_uniform(
      env,
      hard_ids,
      pose_range=hard_pose_range,
      velocity_range=hard_velocity_range,
      asset_cfg=asset_cfg,
    )
    asset = env.scene[asset_cfg.name]
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[hard_ids, :2], dim=1)
    lin_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[hard_ids, :2], dim=1)
    ang_vel = torch.linalg.norm(asset.data.root_link_ang_vel_b[hard_ids, :2], dim=1)
    magnitude = tilt + 0.5 * lin_vel + 0.25 * ang_vel
    _mark_disturbance(
      env,
      hard_ids,
      magnitude=magnitude,
      kind=DISTURBANCE_NEAR_FAILURE_RESET,
      active=False,
    )


class apply_body_impulse_with_history:
  """Wrap ``apply_body_impulse`` and mirror disturbance metadata into env state."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._event = envs_mdp.apply_body_impulse(cfg, env)
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._threshold = cfg.params.get("activation_threshold", 1.0)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    body_point_offset: tuple[float, float, float] | None = None,
    activation_threshold: float = 1.0,
  ) -> None:
    del env_ids, activation_threshold
    prev_active = get_antifall_state(env)["disturbance_active"].clone()
    self._event(
      env,
      None,
      force_range=force_range,
      torque_range=torque_range,
      duration_s=duration_s,
      cooldown_s=cooldown_s,
      asset_cfg=asset_cfg,
      body_point_offset=body_point_offset,
    )

    wrench = self._asset.data.body_external_wrench
    if self._body_ids:
      wrench = wrench[:, self._body_ids]
    force_mag = torch.linalg.norm(wrench[..., :3], dim=-1).amax(dim=1)
    torque_mag = torch.linalg.norm(wrench[..., 3:], dim=-1).amax(dim=1)
    magnitude = force_mag + 0.1 * torque_mag
    active = magnitude > self._threshold

    state = get_antifall_state(env)
    state["disturbance_active"][:] = active
    state["last_disturbance_mag"][active] = magnitude[active]
    state["disturbance_kind"][active] = DISTURBANCE_IMPULSE

    new_ids = (active & ~prev_active).nonzero(as_tuple=False).flatten()
    if len(new_ids) > 0:
      _mark_disturbance(
        env,
        new_ids,
        magnitude=magnitude[new_ids],
        kind=DISTURBANCE_IMPULSE,
        active=True,
      )

    ended = (~active) & prev_active
    if ended.any():
      state["disturbance_active"][ended] = False

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._event.reset(env_ids=env_ids)
    reset_antifall_state(self._env, env_ids)
