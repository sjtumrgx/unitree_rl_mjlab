from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.anti_fall.events import (
  DISTURBANCE_NEAR_FAILURE_RESET,
  get_antifall_state,
  reset_antifall_state,
)
from src.tasks.velocity.mdp.anti_fall.rewards import recovery_phase_mask
from .metrics import _upright_alignment, stable_getup_success_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_PRESETS: tuple[dict[str, object], ...] = (
  {
    "name": "supine",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      # reset_root_state_uniform() samples relative to the standing default root state.
      # Supine has the thickest torso/limb contact stack.  The side-lying z offset
      # penetrates too deeply and creates an upward contact impulse before the
      # policy acts, so supine uses a higher fallen-but-not-standing placement.
      "z": (-0.35, -0.25),
      "roll": (math.pi - 0.3, math.pi + 0.3),
      "pitch": (-0.3, 0.3),
      "yaw": (-math.pi, math.pi),
    },
  },
  {
    "name": "left_side",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (-0.7, -0.6),
      "roll": (math.pi / 2 - 0.25, math.pi / 2 + 0.25),
      "pitch": (-0.35, 0.35),
      "yaw": (-math.pi, math.pi),
    },
  },
  {
    "name": "right_side",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (-0.7, -0.6),
      "roll": (-math.pi / 2 - 0.25, -math.pi / 2 + 0.25),
      "pitch": (-0.35, 0.35),
      "yaw": (-math.pi, math.pi),
    },
  },
  {
    "name": "seated_fall",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (-0.5, -0.4),
      "roll": (-0.2, 0.2),
      "pitch": (math.pi / 2 - 0.35, math.pi / 2 + 0.35),
      "yaw": (-math.pi, math.pi),
    },
  },
)
_DEFAULT_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.25, 0.25),
  "roll": (-0.75, 0.75),
  "pitch": (-0.75, 0.75),
  "yaw": (-0.75, 0.75),
}
_DEFAULT_JOINT_PRESET_TARGETS: dict[str, dict[str, float]] = {
  "supine": {
    "left_hip_pitch_joint": -1.05,
    "left_knee_joint": 1.85,
    "left_ankle_pitch_joint": -0.85,
    "right_hip_pitch_joint": -1.05,
    "right_knee_joint": 1.85,
    "right_ankle_pitch_joint": -0.85,
    "left_shoulder_pitch_joint": 0.75,
    "left_shoulder_roll_joint": 0.28,
    "left_elbow_joint": 1.15,
    "right_shoulder_pitch_joint": 0.75,
    "right_shoulder_roll_joint": -0.28,
    "right_elbow_joint": 1.15,
  },
  "left_side": {
    "left_hip_pitch_joint": -0.95,
    "left_hip_roll_joint": 0.45,
    "left_knee_joint": 1.6,
    "left_ankle_pitch_joint": -0.7,
    "right_hip_pitch_joint": -0.55,
    "right_hip_roll_joint": -0.15,
    "right_knee_joint": 1.1,
    "right_ankle_pitch_joint": -0.45,
    "left_shoulder_pitch_joint": 0.55,
    "left_shoulder_roll_joint": 0.45,
    "left_elbow_joint": 1.05,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.1,
    "right_elbow_joint": 0.9,
  },
  "right_side": {
    "left_hip_pitch_joint": -0.55,
    "left_hip_roll_joint": 0.15,
    "left_knee_joint": 1.1,
    "left_ankle_pitch_joint": -0.45,
    "right_hip_pitch_joint": -0.95,
    "right_hip_roll_joint": -0.45,
    "right_knee_joint": 1.6,
    "right_ankle_pitch_joint": -0.7,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.1,
    "left_elbow_joint": 0.9,
    "right_shoulder_pitch_joint": 0.55,
    "right_shoulder_roll_joint": -0.45,
    "right_elbow_joint": 1.05,
  },
  "seated_fall": {
    "left_hip_pitch_joint": -1.3,
    "left_knee_joint": 2.15,
    "left_ankle_pitch_joint": -1.0,
    "right_hip_pitch_joint": -1.3,
    "right_knee_joint": 2.15,
    "right_ankle_pitch_joint": -1.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_elbow_joint": 1.0,
    "right_shoulder_pitch_joint": 0.4,
    "right_elbow_joint": 1.0,
  },
}


def get_getup_reset_state(
  env: ManagerBasedRlEnv,
  preset_names: Sequence[str] | None = None,
) -> dict[str, object]:
  state = getattr(env, "_getup_reset_state", None)
  if state is None or state["preset_index"].shape[0] != env.num_envs:
    state = {
      "preset_index": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
      "preset_names": tuple(preset_names or ()),
    }
    setattr(env, "_getup_reset_state", state)
  elif preset_names is not None:
    state["preset_names"] = tuple(preset_names)
  return state


def _preset_weights_for_step(
  common_step_counter: int,
  preset_weight_stages: Sequence[dict[str, object]],
) -> tuple[float, ...]:
  weights = tuple(float(weight) for weight in preset_weight_stages[0]["weights"])  # type: ignore[index]
  for stage in preset_weight_stages:
    if common_step_counter >= int(stage["step"]):
      weights = tuple(float(weight) for weight in stage["weights"])  # type: ignore[index]
  return weights


def _ids_from_cfg(ids: list[int] | slice, total_count: int, device: torch.device | str) -> torch.Tensor:
  if isinstance(ids, slice):
    return torch.arange(total_count, device=device, dtype=torch.long)[ids]
  return torch.as_tensor(ids, device=device, dtype=torch.long)


def _validate_reset_joint_targets(
  *,
  selected_preset_names: Sequence[str],
  joint_targets: dict[str, dict[str, float]],
  joint_name_to_index: dict[str, int],
  active_joint_ids: torch.Tensor,
) -> None:
  active_joint_id_set = {int(joint_id) for joint_id in active_joint_ids.detach().cpu().tolist()}
  for preset_name in selected_preset_names:
    targets = joint_targets.get(preset_name)
    if not targets:
      raise ValueError(f"get-up reset preset {preset_name!r} has no joint targets")
    unknown = sorted(name for name in targets if name not in joint_name_to_index)
    if unknown:
      raise ValueError(f"get-up reset preset {preset_name!r} references unknown joints: {unknown}")
    inactive = sorted(
      name
      for name in targets
      if int(joint_name_to_index[name]) not in active_joint_id_set
    )
    if inactive:
      raise ValueError(
        f"get-up reset preset {preset_name!r} references joints outside asset_cfg.joint_ids: {inactive}"
      )


def _stable_getup_success_mask(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  asset,
  body_ids,
  success_height_threshold: float,
  upright_alignment_threshold: float,
  feet_sensor_name: str | None,
  body_sensor_name: str | None,
  hand_sensor_name: str | None = None,
  foot_geom_sensor_name: str | None = None,
  min_feet_contact_count: float,
  max_body_support_count: float,
  max_hand_contact_count: float | None = None,
  min_foot_flatness: float | None = None,
  min_foot_heading_alignment: float | None = None,
  min_foot_geom_contact_spread: float | None = None,
  foot_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  del asset
  if isinstance(body_ids, torch.Tensor):
    torso_asset_cfg = SceneEntityCfg("robot", body_ids=body_ids.detach().cpu().tolist())
  else:
    torso_asset_cfg = SceneEntityCfg("robot", body_ids=body_ids)
  tilt_threshold = math.acos(max(min(float(upright_alignment_threshold), 1.0), -1.0))
  return stable_getup_success_mask(
    env,
    env_ids,
    tilt_threshold=tilt_threshold,
    torso_height_threshold=success_height_threshold,
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
    foot_asset_cfg=foot_asset_cfg or SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    asset_cfg=torso_asset_cfg,
  )


def _contact_count_tensor(
  env: ManagerBasedRlEnv,
  sensor_name: str | None,
  *,
  num_envs: int,
  device: torch.device,
) -> torch.Tensor | None:
  if sensor_name is None:
    return None
  sensor = env.scene.get(sensor_name) if isinstance(getattr(env, "scene", None), dict) else env.scene[sensor_name]
  if sensor is None:
    return None
  found = getattr(getattr(sensor, "data", None), "found", None)
  if found is None:
    return None
  return (found > 0).float().reshape(num_envs, -1).sum(dim=1).to(device=device)


def _assist_decay_milestone_mask(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  asset,
  body_ids,
  success_height_threshold: float,
  upright_alignment_threshold: float,
  feet_sensor_name: str | None,
  body_sensor_name: str | None,
  hand_sensor_name: str | None = None,
  min_feet_contact_count: float,
  max_body_support_count: float,
  max_hand_contact_count: float | None = None,
) -> torch.Tensor:
  torso_height = _relative_torso_height(env, asset, body_ids, env_ids)
  alignment = _upright_alignment(asset.data.projected_gravity_b)[env_ids]
  success = (torso_height >= float(success_height_threshold)) & (
    alignment >= float(upright_alignment_threshold)
  )
  device = torso_height.device
  num_envs = int(asset.data.projected_gravity_b.shape[0])

  feet_count = _contact_count_tensor(env, feet_sensor_name, num_envs=num_envs, device=device)
  if feet_count is not None:
    success &= feet_count[env_ids] >= float(min_feet_contact_count)

  body_count = _contact_count_tensor(env, body_sensor_name, num_envs=num_envs, device=device)
  if body_count is not None:
    success &= body_count[env_ids] <= float(max_body_support_count)

  hand_count = _contact_count_tensor(env, hand_sensor_name, num_envs=num_envs, device=device)
  if hand_count is not None and max_hand_contact_count is not None:
    success &= hand_count[env_ids] <= float(max_hand_contact_count)

  return success


def _mark_recovery_reset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  *,
  disturbance_kind: int = DISTURBANCE_NEAR_FAILURE_RESET,
  disturbance_magnitude: float = 1.0,
) -> None:
  if env_ids.numel() == 0:
    return
  state = get_antifall_state(env)
  state["last_disturbance_step"][env_ids] = int(getattr(env, "common_step_counter", 0))
  state["last_disturbance_mag"][env_ids] = disturbance_magnitude
  state["disturbance_kind"][env_ids] = disturbance_kind
  state["disturbance_active"][env_ids] = False
  state["disturbance_count"][env_ids] = 1


def get_host_getup_curriculum_state(
  env: ManagerBasedRlEnv,
  *,
  initial_force_n: float = 100.0,
  initial_action_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
  """Return HoST-style per-env force/action-scale curriculum buffers."""

  state = getattr(env, "_host_getup_curriculum_state", None)
  if (
    not isinstance(state, dict)
    or state.get("force_n", torch.empty(0, device=env.device)).shape[0] != env.num_envs
  ):
    state = {
      "force_n": torch.full((env.num_envs,), float(initial_force_n), device=env.device),
      "action_rescale": torch.full((env.num_envs,), float(initial_action_scale), device=env.device),
      "max_torso_height": torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
      "episode_success": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
      "episode_force_scale": torch.ones(env.num_envs, dtype=torch.float32, device=env.device),
    }
    setattr(env, "_host_getup_curriculum_state", state)
  if "episode_success" not in state:
    state["episode_success"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if "episode_force_scale" not in state:
    state["episode_force_scale"] = torch.ones(env.num_envs, dtype=torch.float32, device=env.device)
  return state


def _scheduled_no_assist_probability(
  force_n: torch.Tensor,
  *,
  initial_force_n: float,
  min_force_n: float,
  initial_probability: float,
  max_probability: float,
  ramp_start_progress: float = 0.0,
  ramp_end_progress: float = 1.0,
) -> torch.Tensor:
  """Ramp no-assist exposure as the per-env assist curriculum succeeds.

  Starting every resumed episode with a high no-assist probability can erase an
  assisted get-up policy before it has converted those successes to the play
  dynamics.  Tie the probability to the same per-env force decay that records
  stable assisted success: early envs keep enough assisted episodes to maintain
  the skill, while envs that repeatedly succeed and decay the force are pushed
  toward no-assist rollouts.  The ramp window lets training hold a low
  no-assist dose until the assisted policy has already proven repeatable by
  decaying a configurable fraction of the external force.
  """

  initial = min(max(float(initial_probability), 0.0), 1.0)
  maximum = min(max(float(max_probability), 0.0), 1.0)
  if maximum < initial:
    initial, maximum = maximum, initial
  span = max(float(initial_force_n) - float(min_force_n), 1e-6)
  force_decay_progress = torch.clamp(
    (float(initial_force_n) - force_n.float()) / span,
    min=0.0,
    max=1.0,
  )
  ramp_start = min(max(float(ramp_start_progress), 0.0), 1.0)
  ramp_end = min(max(float(ramp_end_progress), 0.0), 1.0)
  if ramp_end < ramp_start:
    ramp_start, ramp_end = ramp_end, ramp_start
  ramp_span = max(ramp_end - ramp_start, 1e-6)
  ramp_progress = torch.clamp((force_decay_progress - ramp_start) / ramp_span, min=0.0, max=1.0)
  return initial + (maximum - initial) * ramp_progress


def _relative_torso_height(
  env: ManagerBasedRlEnv,
  asset,
  body_ids,
  env_ids: torch.Tensor,
) -> torch.Tensor:
  torso_height = asset.data.body_link_pos_w[env_ids][:, body_ids, 2].amax(dim=1)
  env_origins = getattr(env.scene, "env_origins", None)
  if env_origins is None and isinstance(getattr(env, "scene", None), dict):
    env_origins = env.scene.get("env_origins")
  if env_origins is not None:
    torso_height = torso_height - env_origins[env_ids, 2]
  return torso_height


class apply_host_getup_assist_force:
  """HoST-style vertical pull with force/action-scale curriculum.

  HoST does not apply the pulling force during the initial unactuated window and
  gates it by torso orientation (`projected_gravity[:, 2] < -0.8`).  When an
  episode reaches the head-height curriculum threshold, both assist force and
  action rescale decay.  MJLab has a scalar reward path, so this event owns the
  shared curriculum buffers consumed by the action term and metrics.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    self._env = env
    self._device = env.device
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_bodies = len(self._body_ids) if isinstance(self._body_ids, list) else 1
    self._initial_force_n = float(cfg.params.get("initial_force_n", 100.0))
    self._initial_action_scale = float(cfg.params.get("initial_action_scale", 1.0))
    self._success_height_threshold = float(cfg.params.get("success_height_threshold", 0.9))
    self._force_decay_n = float(cfg.params.get("force_decay_n", 20.0))
    self._action_scale_decay = float(cfg.params.get("action_scale_decay", 0.02))
    self._min_force_n = float(cfg.params.get("min_force_n", 0.0))
    self._min_action_scale = float(cfg.params.get("min_action_scale", 0.25))
    self._stable_success_required = bool(cfg.params.get("stable_success_required", True))
    self._assist_decay_requires_strict_success = bool(
      cfg.params.get("assist_decay_requires_strict_success", False)
    )
    self._upright_alignment_threshold = float(cfg.params.get("upright_alignment_threshold", 0.85))
    self._feet_sensor_name = cfg.params.get("feet_sensor_name")
    self._body_sensor_name = cfg.params.get("body_sensor_name")
    self._hand_sensor_name = cfg.params.get("hand_sensor_name")
    self._foot_geom_sensor_name = cfg.params.get("foot_geom_sensor_name")
    self._min_feet_contact_count = float(cfg.params.get("min_feet_contact_count", 1.0))
    self._max_body_support_count = float(cfg.params.get("max_body_support_count", 1.0))
    self._max_hand_contact_count = cfg.params.get("max_hand_contact_count")
    if self._max_hand_contact_count is not None:
      self._max_hand_contact_count = float(self._max_hand_contact_count)
    self._min_foot_flatness = cfg.params.get("min_foot_flatness")
    if self._min_foot_flatness is not None:
      self._min_foot_flatness = float(self._min_foot_flatness)
    self._min_foot_heading_alignment = cfg.params.get("min_foot_heading_alignment")
    if self._min_foot_heading_alignment is not None:
      self._min_foot_heading_alignment = float(self._min_foot_heading_alignment)
    self._min_foot_geom_contact_spread = cfg.params.get("min_foot_geom_contact_spread")
    if self._min_foot_geom_contact_spread is not None:
      self._min_foot_geom_contact_spread = float(self._min_foot_geom_contact_spread)
    self._foot_asset_cfg = cfg.params.get(
      "foot_asset_cfg",
      SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")),
    )
    self._no_assist_probability = float(cfg.params.get("no_assist_probability", 0.0))
    self._no_assist_probability_initial = float(
      cfg.params.get(
        "no_assist_probability_initial",
        min(max(self._no_assist_probability, 0.0), 1.0),
      )
    )
    self._no_assist_ramp_start_progress = float(cfg.params.get("no_assist_ramp_start_progress", 0.0))
    self._no_assist_ramp_end_progress = float(cfg.params.get("no_assist_ramp_end_progress", 1.0))
    get_host_getup_curriculum_state(
      env,
      initial_force_n=self._initial_force_n,
      initial_action_scale=self._initial_action_scale,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    initial_force_n: float = 100.0,
    initial_action_scale: float = 1.0,
    success_height_threshold: float = 0.9,
    force_decay_n: float = 20.0,
    action_scale_decay: float = 0.02,
    min_force_n: float = 0.0,
    min_action_scale: float = 0.25,
    unactuated_timesteps: int = 30,
    orientation_projected_gravity_z_max: float = -0.8,
    no_orientation_gate: bool = False,
    stable_success_required: bool = True,
    upright_alignment_threshold: float = 0.85,
    feet_sensor_name: str | None = None,
    body_sensor_name: str | None = None,
    hand_sensor_name: str | None = None,
    foot_geom_sensor_name: str | None = None,
    min_feet_contact_count: float = 1.0,
    max_body_support_count: float = 1.0,
    max_hand_contact_count: float | None = None,
    min_foot_flatness: float | None = None,
    min_foot_heading_alignment: float | None = None,
    min_foot_geom_contact_spread: float | None = None,
    foot_asset_cfg: SceneEntityCfg | None = None,
    taper_start_height: float = 0.45,
    taper_end_height: float = 0.70,
    no_assist_probability: float = 0.0,
    no_assist_probability_initial: float | None = None,
    no_assist_ramp_start_progress: float = 0.0,
    no_assist_ramp_end_progress: float = 1.0,
    recovery_phase_only: bool = False,
    fallen_height_threshold: float = 0.35,
    fallen_tilt_threshold: float = 0.75,
    recovery_window_s: float = 2.0,
    include_disturbance_window: bool = True,
    include_near_failure_reset_window: bool = True,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del (
      no_assist_probability,
      no_assist_probability_initial,
      no_assist_ramp_start_progress,
      no_assist_ramp_end_progress,
    )
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    state = get_host_getup_curriculum_state(
      env,
      initial_force_n=initial_force_n,
      initial_action_scale=initial_action_scale,
    )
    asset = env.scene[asset_cfg.name]
    torso_height = _relative_torso_height(env, asset, asset_cfg.body_ids, env_ids)
    state["max_torso_height"][env_ids] = torch.maximum(
      state["max_torso_height"][env_ids],
      torso_height,
    )
    assist_decay_requires_strict_success = bool(
      getattr(self, "_assist_decay_requires_strict_success", False)
    )
    if stable_success_required and assist_decay_requires_strict_success:
      succeeded_now = _stable_getup_success_mask(
        env,
        env_ids,
        asset=asset,
        body_ids=asset_cfg.body_ids,
        success_height_threshold=success_height_threshold,
        upright_alignment_threshold=upright_alignment_threshold,
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
      )
    elif stable_success_required:
      succeeded_now = _assist_decay_milestone_mask(
        env,
        env_ids,
        asset=asset,
        body_ids=asset_cfg.body_ids,
        success_height_threshold=success_height_threshold,
        upright_alignment_threshold=upright_alignment_threshold,
        feet_sensor_name=feet_sensor_name,
        body_sensor_name=body_sensor_name,
        hand_sensor_name=hand_sensor_name,
        min_feet_contact_count=min_feet_contact_count,
        max_body_support_count=max_body_support_count,
        max_hand_contact_count=max_hand_contact_count,
      )
    else:
      succeeded_now = torso_height > float(success_height_threshold)
    state["episode_success"][env_ids] |= succeeded_now

    episode_length = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device))
    past_startup = episode_length[env_ids] > int(unactuated_timesteps)
    if no_orientation_gate:
      oriented = torch.ones_like(past_startup, dtype=torch.bool, device=env.device)
    else:
      oriented = asset.data.projected_gravity_b[env_ids, 2] < float(orientation_projected_gravity_z_max)
    # Once an episode has reached stable get-up success, stop applying the
    # external wrench for that env immediately.  Otherwise the policy can keep
    # receiving standing rewards while being held upright by the curriculum
    # force, which does not transfer to the play/no-assist environment.
    episode_force_scale = state["episode_force_scale"][env_ids]
    active = (
      past_startup
      & oriented
      & (state["force_n"][env_ids] > 0.0)
      & (episode_force_scale > 0.0)
      & ~state["episode_success"][env_ids]
    )
    if recovery_phase_only:
      phase_mask = recovery_phase_mask(
        env,
        fallen_height_threshold=fallen_height_threshold,
        fallen_tilt_threshold=fallen_tilt_threshold,
        window_s=recovery_window_s,
        include_disturbance_window=include_disturbance_window,
        include_near_failure_reset_window=include_near_failure_reset_window,
        asset_cfg=asset_cfg,
      )[env_ids]
      active = active & phase_mask
    taper_span = max(float(taper_end_height) - float(taper_start_height), 1e-6)
    assist_fraction = torch.clamp(
      (float(taper_end_height) - torso_height) / taper_span,
      min=0.0,
      max=1.0,
    )

    forces = torch.zeros((env_ids.numel(), self._num_bodies, 3), device=env.device)
    torques = torch.zeros_like(forces)
    forces[active, :, 2] = (
      state["force_n"][env_ids][active] * episode_force_scale[active] * assist_fraction[active]
    ).unsqueeze(1)
    asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    state = get_host_getup_curriculum_state(
      self._env,
      initial_force_n=self._initial_force_n,
      initial_action_scale=self._initial_action_scale,
    )
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      env_ids_t = torch.arange(self._env.num_envs, device=self._device, dtype=torch.long)[env_ids]
    else:
      env_ids_t = env_ids.to(device=self._device, dtype=torch.long)
    if env_ids_t.numel() == 0:
      return

    height_success = state["max_torso_height"][env_ids_t] > self._success_height_threshold
    succeeded = height_success
    if self._stable_success_required and self._assist_decay_requires_strict_success:
      current_stable_success = height_success & _stable_getup_success_mask(
        self._env,
        env_ids_t,
        asset=self._asset,
        body_ids=self._body_ids,
        success_height_threshold=self._success_height_threshold,
        upright_alignment_threshold=self._upright_alignment_threshold,
        feet_sensor_name=self._feet_sensor_name,
        body_sensor_name=self._body_sensor_name,
        hand_sensor_name=self._hand_sensor_name,
        foot_geom_sensor_name=self._foot_geom_sensor_name,
        min_feet_contact_count=self._min_feet_contact_count,
        max_body_support_count=self._max_body_support_count,
        max_hand_contact_count=self._max_hand_contact_count,
        min_foot_flatness=self._min_foot_flatness,
        min_foot_heading_alignment=self._min_foot_heading_alignment,
        min_foot_geom_contact_spread=self._min_foot_geom_contact_spread,
        foot_asset_cfg=self._foot_asset_cfg,
      )
      succeeded = state["episode_success"][env_ids_t] | current_stable_success
    elif self._stable_success_required:
      current_milestone_success = height_success & _assist_decay_milestone_mask(
        self._env,
        env_ids_t,
        asset=self._asset,
        body_ids=self._body_ids,
        success_height_threshold=self._success_height_threshold,
        upright_alignment_threshold=self._upright_alignment_threshold,
        feet_sensor_name=self._feet_sensor_name,
        body_sensor_name=self._body_sensor_name,
        hand_sensor_name=self._hand_sensor_name,
        min_feet_contact_count=self._min_feet_contact_count,
        max_body_support_count=self._max_body_support_count,
        max_hand_contact_count=self._max_hand_contact_count,
      )
      succeeded = state["episode_success"][env_ids_t] | current_milestone_success
    if torch.any(succeeded):
      selected = env_ids_t[succeeded]
      state["force_n"][selected] = torch.clamp(
        state["force_n"][selected] - self._force_decay_n,
        min=self._min_force_n,
      )
      state["action_rescale"][selected] = torch.clamp(
        state["action_rescale"][selected] - self._action_scale_decay,
        min=self._min_action_scale,
      )
    state["max_torso_height"][env_ids_t] = 0.0
    state["episode_success"][env_ids_t] = False
    no_assist_probability = _scheduled_no_assist_probability(
      state["force_n"][env_ids_t],
      initial_force_n=self._initial_force_n,
      min_force_n=self._min_force_n,
      initial_probability=self._no_assist_probability_initial,
      max_probability=self._no_assist_probability,
      ramp_start_progress=self._no_assist_ramp_start_progress,
      ramp_end_progress=self._no_assist_ramp_end_progress,
    )
    if torch.any(no_assist_probability > 0.0):
      keep_assist = torch.rand(env_ids_t.numel(), device=self._device) >= no_assist_probability
      state["episode_force_scale"][env_ids_t] = keep_assist.to(dtype=torch.float32)
    else:
      state["episode_force_scale"][env_ids_t] = 1.0

    forces = torch.zeros((env_ids_t.numel(), self._num_bodies, 3), device=self._device)
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids_t, body_ids=self._body_ids)


class apply_getup_assist_force:
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._env = env
    self._device = env.device
    self._asset = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_bodies = len(self._body_ids) if isinstance(self._body_ids, list) else 1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_n: float,
    activation_height: float,
    alignment_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    asset = env.scene[asset_cfg.name]
    torso_height = asset.data.body_link_pos_w[env_ids][:, asset_cfg.body_ids, 2].amax(dim=1)
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is None and isinstance(getattr(env, "scene", None), dict):
      env_origins = env.scene.get("env_origins")
    if env_origins is not None:
      torso_height = torso_height - env_origins[env_ids, 2]
    alignment = _upright_alignment(asset.data.projected_gravity_b[env_ids])
    active = (torso_height < activation_height) & (alignment >= alignment_threshold)
    num_envs = env_ids.numel()
    forces = torch.zeros((num_envs, self._num_bodies, 3), device=env.device)
    torques = torch.zeros_like(forces)
    forces[active, :, 2] = force_n
    asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      env_ids = torch.arange(self._env.num_envs, device=self._device, dtype=torch.long)[env_ids]
    forces = torch.zeros((len(env_ids), self._num_bodies, 3), device=self._device)
    torques = torch.zeros_like(forces)
    self._asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids, body_ids=self._body_ids)


def getup_assist_force_n(
  env: ManagerBasedRlEnv,
  initial_force_n: float = 100.0,
) -> torch.Tensor:
  state = get_host_getup_curriculum_state(env, initial_force_n=initial_force_n)
  return state["force_n"]


def getup_action_rescale(
  env: ManagerBasedRlEnv,
  initial_action_scale: float = 1.0,
) -> torch.Tensor:
  state = get_host_getup_curriculum_state(env, initial_action_scale=initial_action_scale)
  return state["action_rescale"]


def reset_root_state_from_presets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  presets: Sequence[dict[str, object]] | None = None,
  preset_weight_stages: Sequence[dict[str, object]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    ids = env_ids.to(device=env.device, dtype=torch.long)
  if ids.numel() == 0:
    return

  reset_antifall_state(env, ids)
  preset_list = tuple(_DEFAULT_PRESETS if presets is None else presets)
  vel_range = _DEFAULT_VELOCITY_RANGE if velocity_range is None else velocity_range
  if preset_weight_stages:
    weights = torch.tensor(
      _preset_weights_for_step(int(getattr(env, "common_step_counter", 0)), preset_weight_stages),
      device=env.device,
      dtype=torch.float32,
    )
    weights = weights / torch.clamp(weights.sum(), min=1e-6)
    preset_indices = torch.multinomial(weights, ids.numel(), replacement=True)
  else:
    preset_indices = torch.randint(len(preset_list), (ids.numel(),), device=env.device)
  state = get_getup_reset_state(
    env,
    preset_names=tuple(str(preset["name"]) for preset in preset_list),
  )
  state["preset_index"][ids] = preset_indices
  for preset_idx, preset in enumerate(preset_list):
    selected = ids[preset_indices == preset_idx]
    if selected.numel() == 0:
      continue
    envs_mdp.reset_root_state_uniform(
      env,
      selected,
      pose_range=preset["pose_range"],
      velocity_range=vel_range,
      asset_cfg=asset_cfg,
    )
    _mark_recovery_reset(env, selected)


def reset_joints_from_presets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_noise_range: tuple[float, float] = (-0.05, 0.05),
  velocity_range: tuple[float, float] = (-0.5, 0.5),
  preset_joint_targets: dict[str, dict[str, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    ids = env_ids.to(device=env.device, dtype=torch.long)
  if ids.numel() == 0:
    return

  asset = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[ids][:, asset_cfg.joint_ids].clone()
  joint_vel = default_joint_vel[ids][:, asset_cfg.joint_ids].clone()

  state = get_getup_reset_state(env)
  preset_names = tuple(state["preset_names"])
  preset_indices: torch.Tensor = state["preset_index"][ids]
  joint_targets = _DEFAULT_JOINT_PRESET_TARGETS if preset_joint_targets is None else preset_joint_targets
  joint_name_to_index = {name: idx for idx, name in enumerate(asset.joint_names)}
  active_joint_ids = _ids_from_cfg(asset_cfg.joint_ids, len(asset.joint_names), env.device)
  selected_preset_names = tuple(
    preset_names[int(preset_idx)]
    for preset_idx in torch.unique(preset_indices).detach().cpu().tolist()
    if int(preset_idx) >= 0 and int(preset_idx) < len(preset_names)
  )
  if len(selected_preset_names) != int(torch.unique(preset_indices).numel()):
    raise ValueError(
      f"get-up reset preset index is outside preset_names: "
      f"indices={torch.unique(preset_indices).detach().cpu().tolist()}, preset_names={preset_names}"
    )
  _validate_reset_joint_targets(
    selected_preset_names=selected_preset_names,
    joint_targets=joint_targets,
    joint_name_to_index=joint_name_to_index,
    active_joint_ids=active_joint_ids,
  )

  active_joint_position = {int(joint_id): pos for pos, joint_id in enumerate(active_joint_ids.detach().cpu().tolist())}
  for preset_idx, preset_name in enumerate(preset_names):
    selected_mask = preset_indices == preset_idx
    if not torch.any(selected_mask):
      continue
    selected_rows = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    for joint_name, target in joint_targets[preset_name].items():
      joint_idx = int(joint_name_to_index[joint_name])
      joint_pos[selected_rows, active_joint_position[joint_idx]] = target

  if position_noise_range != (0.0, 0.0):
    joint_pos += envs_mdp.sample_uniform(*position_noise_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  if velocity_range != (0.0, 0.0):
    joint_vel += envs_mdp.sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids, joint_ids=joint_ids)
