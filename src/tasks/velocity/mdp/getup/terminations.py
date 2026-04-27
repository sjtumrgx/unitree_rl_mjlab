from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.mdp.anti_fall.events import disturbance_window_mask
from src.tasks.velocity.mdp.terminations import illegal_contact

from .metrics import _relative_body_height, _upright_alignment

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_TORSO_ASSET_CFG = SceneEntityCfg("robot", body_names=("torso_link",))


def _getup_progress(
  env: ManagerBasedRlEnv,
  *,
  min_height: float,
  target_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  facing_up = torch.clamp(_upright_alignment(asset.data.projected_gravity_b), min=0.0, max=1.0)
  body_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  torso_height = _relative_body_height(env, body_heights).amax(dim=1)
  height_progress = torch.clamp(
    (torso_height - min_height) / max(target_height - min_height, 1e-6),
    min=0.0,
    max=1.0,
  )
  return height_progress * facing_up


class stalled_getup_progress:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._best_progress = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    self._elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    min_steps_before_check: int = 50,
    progress_threshold: float = 0.2,
    recovery_grace_s: float = 0.0,
    min_height: float = 0.12,
    target_height: float = 0.55,
    asset_cfg: SceneEntityCfg = _TORSO_ASSET_CFG,
  ) -> torch.Tensor:
    if recovery_grace_s > 0.0:
      in_recovery_grace = disturbance_window_mask(env, recovery_grace_s)
    else:
      in_recovery_grace = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    progress = _getup_progress(
      env,
      min_height=min_height,
      target_height=target_height,
      asset_cfg=asset_cfg,
    )
    self._elapsed_steps = torch.where(
      in_recovery_grace,
      self._elapsed_steps,
      self._elapsed_steps + 1,
    )
    self._best_progress = torch.where(
      in_recovery_grace,
      self._best_progress,
      torch.maximum(self._best_progress, progress),
    )
    return (~in_recovery_grace) & (self._elapsed_steps >= min_steps_before_check) & (
      self._best_progress < progress_threshold
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._best_progress[env_ids] = 0.0
    self._elapsed_steps[env_ids] = 0


class tolerant_illegal_contact_during_recovery:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._bad_contact_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
    bad_contact_time_threshold_s: float = 0.1,
    grace_period_s: float = 0.0,
    recovery_grace_s: float = 0.0,
  ) -> torch.Tensor:
    if recovery_grace_s > 0.0:
      in_recovery_grace = disturbance_window_mask(env, recovery_grace_s)
    else:
      in_recovery_grace = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    bad_contact = illegal_contact(env, sensor_name=sensor_name, force_threshold=force_threshold)
    grace_steps = max(0, math.ceil(grace_period_s / max(env.step_dt, 1e-6)))
    bad_contact_steps = max(1, math.ceil(bad_contact_time_threshold_s / max(env.step_dt, 1e-6)))
    eligible = env.episode_length_buf >= grace_steps
    self._bad_contact_steps = torch.where(
      in_recovery_grace,
      torch.zeros_like(self._bad_contact_steps),
      torch.where(
        eligible & bad_contact,
        self._bad_contact_steps + 1,
        torch.zeros_like(self._bad_contact_steps),
      ),
    )
    return (~in_recovery_grace) & (self._bad_contact_steps >= bad_contact_steps)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._bad_contact_steps[env_ids] = 0
