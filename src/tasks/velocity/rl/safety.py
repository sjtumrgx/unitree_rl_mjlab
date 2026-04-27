"""Safety wrappers for RL environment boundaries."""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper


def sanitize_policy_actions(
  actions: torch.Tensor,
  *,
  clip_actions: float | None,
) -> torch.Tensor:
  """Return finite policy actions with the same clipping contract as the env.

  ``torch.clamp`` bounds ``+/-inf`` but preserves ``NaN``.  Once PPO produces a
  single non-finite action, MJLab stores it in ``last_action`` and RSL-RL aborts
  on the next actor observation.  Sanitize before the environment sees the
  tensor so action targets, action-rate rewards, and action observations all
  stay finite.
  """

  if clip_actions is None:
    return torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

  finite_actions = torch.nan_to_num(
    actions,
    nan=0.0,
    posinf=float(clip_actions),
    neginf=-float(clip_actions),
  )
  return torch.clamp(finite_actions, -clip_actions, clip_actions)


class FiniteActionRslRlVecEnvWrapper(RslRlVecEnvWrapper):
  """RSL-RL wrapper variant that never forwards non-finite actions."""

  def step(
    self,
    actions: torch.Tensor,
  ):
    return super().step(
      sanitize_policy_actions(actions, clip_actions=self.clip_actions)
    )
