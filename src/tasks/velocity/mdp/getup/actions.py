"""HoST-compatible action terms for get-up tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class HostRelativeJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with HoST get-up delta-from-current semantics.

  HoST computes position targets as ``current_joint_pos + action_rescale * action``.
  The base MJLab ``JointPositionActionCfg(use_default_offset=True)`` instead
  targets ``default_joint_pos + scale * action``.  Get-up starts from fallen and
  side-lying states, so preserving the current-pose reference frame is part of
  the task contract rather than a tuning detail.
  """

  use_default_offset: bool = False
  unactuated_timesteps: int = 0
  max_delta: float | None = None
  """Optional per-step delta clamp for current-pose get-up actions."""

  def build(self, env: ManagerBasedRlEnv) -> HostRelativeJointPositionAction:
    return HostRelativeJointPositionAction(self, env)


class HostRelativeJointPositionAction(JointPositionAction):
  """Apply policy actions as deltas from the current joint position."""

  cfg: HostRelativeJointPositionActionCfg

  def process_actions(self, actions):
    super().process_actions(actions)
    startup_steps = max(0, int(self.cfg.unactuated_timesteps))
    if startup_steps > 0:
      episode_length = getattr(self._env, "episode_length_buf", None)
      if episode_length is not None:
        active = (episode_length >= startup_steps).to(
          device=self._processed_actions.device,
          dtype=self._processed_actions.dtype,
        )
        self._processed_actions = self._processed_actions * active.unsqueeze(1)

    state = getattr(self._env, "_host_getup_curriculum_state", None)
    if isinstance(state, dict) and "action_rescale" in state:
      action_rescale = state["action_rescale"].to(
        device=self._processed_actions.device,
        dtype=self._processed_actions.dtype,
      )
      if action_rescale.ndim == 1:
        action_rescale = action_rescale.unsqueeze(1)
      self._processed_actions = self._processed_actions * action_rescale
    if self.cfg.max_delta is not None:
      max_delta = float(self.cfg.max_delta)
      if max_delta <= 0.0:
        raise ValueError("HostRelativeJointPositionActionCfg.max_delta must be positive when set")
      self._processed_actions = self._processed_actions.clamp(-max_delta, max_delta)

  def apply_actions(self) -> None:
    current_joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = current_joint_pos + self._processed_actions - encoder_bias
    setattr(self._env, "_host_getup_joint_position_delta", self._processed_actions.detach().clone())
    setattr(self._env, "_host_getup_joint_position_target", target.detach().clone())
    setattr(self._env, "_host_getup_joint_target_ids", self._target_ids.detach().clone())
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
