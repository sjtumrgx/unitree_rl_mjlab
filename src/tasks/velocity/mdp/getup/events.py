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
from .metrics import _upright_alignment

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
      # Fallen starts therefore need a large negative z-offset to place the body on terrain
      # rather than spawning in the air at standing height and wasting the first recovery
      # steps on a passive fall.
      "z": (-0.7, -0.6),
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
    "waist_roll_joint": 0.35,
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
    "waist_roll_joint": -0.35,
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
    "waist_pitch_joint": 0.35,
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

  for preset_idx, preset_name in enumerate(preset_names):
    selected_mask = preset_indices == preset_idx
    if not torch.any(selected_mask):
      continue
    selected_rows = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    for joint_name, target in joint_targets.get(preset_name, {}).items():
      joint_idx = joint_name_to_index.get(joint_name)
      if joint_idx is None:
        continue
      joint_pos[selected_rows, joint_idx] = target

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
