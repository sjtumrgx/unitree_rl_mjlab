from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


class tolerant_illegal_contact:
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
  ) -> torch.Tensor:
    bad_contact = illegal_contact(env, sensor_name=sensor_name, force_threshold=force_threshold)
    grace_steps = max(0, math.ceil(grace_period_s / max(env.step_dt, 1e-6)))
    bad_contact_steps = max(1, math.ceil(bad_contact_time_threshold_s / max(env.step_dt, 1e-6)))
    eligible = env.episode_length_buf >= grace_steps
    self._bad_contact_steps = torch.where(
      eligible & bad_contact,
      self._bad_contact_steps + 1,
      torch.zeros_like(self._bad_contact_steps),
    )
    return self._bad_contact_steps >= bad_contact_steps

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._bad_contact_steps[env_ids] = 0
