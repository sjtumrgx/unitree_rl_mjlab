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

  def build(self, env: ManagerBasedRlEnv) -> HostRelativeJointPositionAction:
    return HostRelativeJointPositionAction(self, env)


class HostRelativeJointPositionAction(JointPositionAction):
  """Apply policy actions as deltas from the current joint position."""

  def apply_actions(self) -> None:
    current_joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = current_joint_pos + self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
