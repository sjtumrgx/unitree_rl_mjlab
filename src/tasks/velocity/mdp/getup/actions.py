"""HoST-compatible action terms for get-up tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from src.tasks.velocity.mdp.anti_fall.events import disturbance_window_mask

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

  def __init__(self, cfg: HostRelativeJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._effective_actions = self._processed_actions.clone()
    self._prev_effective_actions = self._processed_actions.clone()
    self._prev_prev_effective_actions = self._processed_actions.clone()
    self._publish_effective_action_history()

  @property
  def effective_action(self):
    """Action deltas that are actually applied after HoST get-up gating."""

    return self._effective_actions

  @property
  def prev_effective_action(self):
    """Previous applied HoST get-up action deltas."""

    return self._prev_effective_actions

  @property
  def prev_prev_effective_action(self):
    """Applied HoST get-up action deltas from two policy steps ago."""

    return self._prev_prev_effective_actions

  def _publish_effective_action_history(self) -> None:
    setattr(self._env, "_host_getup_effective_action", self._effective_actions)
    setattr(self._env, "_host_getup_prev_effective_action", self._prev_effective_actions)
    setattr(self._env, "_host_getup_prev_prev_effective_action", self._prev_prev_effective_actions)

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
      episode_force_scale = state.get("episode_force_scale")
      if episode_force_scale is not None:
        no_assist_episode = episode_force_scale.to(
          device=self._processed_actions.device,
          dtype=self._processed_actions.dtype,
        ) <= 0.0
        # No-assist curriculum episodes are the transfer bridge to play mode:
        # they must remove both the external force and the train-only action
        # down-scaling so the policy experiences the same dynamics/action scale
        # as evaluation.  Keep assisted episodes on the decaying HoST scale.
        action_rescale = torch.where(no_assist_episode, torch.ones_like(action_rescale), action_rescale)
      if action_rescale.ndim == 1:
        action_rescale = action_rescale.unsqueeze(1)
      self._processed_actions = self._processed_actions * action_rescale
    if self.cfg.max_delta is not None:
      max_delta = float(self.cfg.max_delta)
      if max_delta <= 0.0:
        raise ValueError("HostRelativeJointPositionActionCfg.max_delta must be positive when set")
      self._processed_actions = self._processed_actions.clamp(-max_delta, max_delta)
    self._prev_prev_effective_actions[:] = self._prev_effective_actions
    self._prev_effective_actions[:] = self._effective_actions
    self._effective_actions[:] = self._processed_actions
    self._publish_effective_action_history()

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
    self._effective_actions[env_ids] = 0.0
    self._prev_effective_actions[env_ids] = 0.0
    self._prev_prev_effective_actions[env_ids] = 0.0
    self._publish_effective_action_history()


@dataclass(kw_only=True)
class RecoveryHybridJointPositionActionCfg(JointPositionActionCfg):
  """Default-offset walking action with current-pose recovery deltas.

  AntiFall-GetUp is warm-started from the walking AntiFall Stage4b policy, whose
  action contract is the normal MJLab/default-joint-position target.  Replacing
  that contract globally with HoST current-pose deltas destroys the walking
  policy before recovery learning starts.  This hybrid keeps the warm-start
  action semantics while the robot is upright, then switches to HoST-like
  current-pose deltas only during a disturbance/recovery window or when the
  state is already fallen.
  """

  use_default_offset: bool = True
  recovery_use_default_offset: bool = False
  recovery_window_s: float = 2.0
  fallen_height_threshold: float = 0.35
  fallen_tilt_threshold: float = 0.75
  recovery_action_scale: float = 1.0
  max_delta: float | None = None
  """Optional clamp for current-pose recovery deltas."""

  def build(self, env: ManagerBasedRlEnv) -> RecoveryHybridJointPositionAction:
    return RecoveryHybridJointPositionAction(self, env)


class RecoveryHybridJointPositionAction(JointPositionAction):
  """Phase-gated joint-position action for walking plus get-up recovery."""

  cfg: RecoveryHybridJointPositionActionCfg

  def __init__(self, cfg: RecoveryHybridJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._walking_targets = self._processed_actions.clone()
    self._recovery_deltas = self._processed_actions.clone()
    self._effective_actions = self._processed_actions.clone()
    self._prev_effective_actions = self._processed_actions.clone()
    self._prev_prev_effective_actions = self._processed_actions.clone()
    self._publish_effective_action_history()

  @property
  def effective_action(self):
    """Raw walking actions or clamped current-pose recovery deltas."""

    return self._effective_actions

  @property
  def prev_effective_action(self):
    return self._prev_effective_actions

  @property
  def prev_prev_effective_action(self):
    return self._prev_prev_effective_actions

  def _publish_effective_action_history(self) -> None:
    setattr(self._env, "_host_getup_effective_action", self._effective_actions)
    setattr(self._env, "_host_getup_prev_effective_action", self._prev_effective_actions)
    setattr(self._env, "_host_getup_prev_prev_effective_action", self._prev_prev_effective_actions)

  def _relative_torso_height(self) -> torch.Tensor:
    torso_height = self._entity.data.root_link_pos_w[:, 2]
    body_pos = getattr(self._entity.data, "body_link_pos_w", None)
    body_names = list(getattr(self._entity, "body_names", ()))
    if body_pos is not None and "torso_link" in body_names:
      torso_id = body_names.index("torso_link")
      torso_height = body_pos[:, torso_id, 2]
    else:
      find_bodies = getattr(self._entity, "find_bodies", None)
      if callable(find_bodies) and body_pos is not None:
        try:
          ids, _ = find_bodies("torso_link")
          if ids:
            torso_height = body_pos[:, int(ids[0]), 2]
        except Exception:
          pass
    env_origins = getattr(self._env.scene, "env_origins", None)
    if env_origins is None and isinstance(getattr(self._env, "scene", None), dict):
      env_origins = self._env.scene.get("env_origins")
    if env_origins is not None:
      torso_height = torso_height - env_origins[:, 2]
    return torso_height

  def _fallen_mask(self) -> torch.Tensor:
    torso_height = self._relative_torso_height()
    projected_gravity = self._entity.data.projected_gravity_b
    tilt = torch.linalg.norm(projected_gravity[:, :2], dim=1)
    return (torso_height < float(self.cfg.fallen_height_threshold)) | (
      tilt > float(self.cfg.fallen_tilt_threshold)
    )

  def _recovery_mask(self) -> torch.Tensor:
    """Use current-pose recovery deltas only after the robot is actually fallen.

    The anti-fall task keeps a BFM-style disturbance/recovery window for rewards,
    metrics, and stall guards, but the warm-started walking actor must retain its
    default-offset action contract while it is still upright inside a push
    window.  Switching every upright post-push step to current-pose deltas makes
    the inherited Stage4b walking policy behave like a different controller and
    destroys pre-disturbance tracking before recovery learning can help.
    """

    return self._fallen_mask()

  def _compute_recovery_deltas(self, actions: torch.Tensor) -> torch.Tensor:
    deltas = actions * float(self.cfg.recovery_action_scale)
    if self.cfg.max_delta is not None:
      max_delta = float(self.cfg.max_delta)
      if max_delta <= 0.0:
        raise ValueError("RecoveryHybridJointPositionActionCfg.max_delta must be positive when set")
      deltas = deltas.clamp(-max_delta, max_delta)
    return deltas

  def process_actions(self, actions):
    super().process_actions(actions)
    self._walking_targets[:] = self._processed_actions
    self._recovery_deltas[:] = self._compute_recovery_deltas(actions)

    recovery_mask = self._recovery_mask().unsqueeze(1)
    effective = torch.where(recovery_mask, self._recovery_deltas, self._raw_actions)
    self._prev_prev_effective_actions[:] = self._prev_effective_actions
    self._prev_effective_actions[:] = self._effective_actions
    self._effective_actions[:] = effective
    self._publish_effective_action_history()

  def apply_actions(self) -> None:
    current_joint_pos = self._entity.data.joint_pos[:, self._target_ids]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    recovery_mask = self._recovery_mask().unsqueeze(1)
    recovery_target = current_joint_pos + self._recovery_deltas
    target = torch.where(recovery_mask, recovery_target, self._walking_targets)
    written_delta = target - current_joint_pos
    setattr(self._env, "_host_getup_joint_position_delta", written_delta.detach().clone())
    setattr(self._env, "_host_getup_joint_position_target", target.detach().clone())
    setattr(self._env, "_host_getup_joint_target_ids", self._target_ids.detach().clone())
    self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self._walking_targets[env_ids] = 0.0
    self._recovery_deltas[env_ids] = 0.0
    self._effective_actions[env_ids] = 0.0
    self._prev_effective_actions[env_ids] = 0.0
    self._prev_prev_effective_actions[env_ids] = 0.0
    self._publish_effective_action_history()
