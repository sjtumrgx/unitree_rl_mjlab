"""Script to train RL agent with RSL-RL."""

import ast
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

import tyro
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from src.tasks.velocity.rl.safety import FiniteActionRslRlVecEnvWrapper

_G1_23DOF_ACTION_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)
_G1_ACTION_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
_G1_GETUP_23DOF_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_rubber_hand",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_rubber_hand",
)
_G1_ANTIFALL_29DOF_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class _ObsTermLayout:
  name: str
  feature_names: tuple[str, ...]
  history: int

  @property
  def width(self) -> int:
    return len(self.feature_names) * self.history


@dataclass(frozen=True)
class _ObservationProjection:
  stats_source_by_target: tuple[int | None, ...]
  weight_source_by_target: tuple[int | None, ...]


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  resume_checkpoint_path: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: str | None = "[0]"
  getup_terrain: str | None = None
  actor_only_resume: bool = False
  policy_only_resume: bool = False
  reset_actor_std_on_resume: bool = False

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _normalize_gpu_ids_cli_args(args: list[str]) -> list[str]:
  """Normalize legacy `--gpu-ids 0 1` usage into a single list token.

  Tyro handles a quoted list such as `--gpu-ids "[0,1]"` cleanly. This helper
  preserves that new format while remaining backwards-compatible with the older
  multi-token form.
  """
  normalized: list[str] = []
  index = 0
  while index < len(args):
    token = args[index]
    if token != "--gpu-ids":
      normalized.append(token)
      index += 1
      continue

    normalized.append(token)
    index += 1

    values: list[str] = []
    while index < len(args) and not args[index].startswith("--"):
      values.append(args[index])
      index += 1

    if len(values) <= 1:
      normalized.extend(values)
      continue

    normalized.append(f"[{','.join(values)}]")

  return normalized


def _parse_gpu_ids_arg(gpu_ids: str | None) -> list[int] | Literal["all"] | None:
  """Parse CLI GPU selection into the format expected by `select_gpus`."""
  if gpu_ids is None:
    return None

  value = gpu_ids.strip()
  if value == "":
    raise ValueError("`--gpu-ids` cannot be empty.")

  lowered = value.lower()
  if lowered == "all":
    return "all"
  if lowered in {"none", "cpu"}:
    return None

  if value[0] in "[(":
    parsed = ast.literal_eval(value)
  elif "," in value:
    parsed = [part.strip() for part in value.split(",") if part.strip()]
  else:
    parsed = [value]

  if isinstance(parsed, int):
    parsed = [parsed]
  elif isinstance(parsed, tuple):
    parsed = list(parsed)

  if not isinstance(parsed, list):
    raise ValueError(
      "`--gpu-ids` must be a list like \"[0,1]\", a comma-separated string like "
      "\"0,1\", a single GPU id, \"all\", or \"cpu\"."
    )

  normalized: list[int] = []
  for item in parsed:
    if isinstance(item, int):
      gpu_id = item
    elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
      gpu_id = int(item.strip())
    else:
      raise ValueError(
        "`--gpu-ids` entries must be integers, e.g. --gpu-ids \"[0,1]\"."
      )
    if gpu_id < 0:
      raise ValueError("`--gpu-ids` entries must be non-negative integers.")
    normalized.append(gpu_id)

  return normalized


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.resume_checkpoint_path is not None:
      # Load an explicit local checkpoint path.  This is intentionally separate
      # from get_checkpoint_path(), whose run/checkpoint regex lookup is scoped
      # to the current experiment log root and cannot warm-start from another
      # experiment such as g1_getup -> g1_getup_amp.
      resume_path = Path(cfg.resume_checkpoint_path).expanduser()
      if not resume_path.is_absolute():
        resume_path = Path.cwd() / resume_path
      if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = FiniteActionRslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if cfg.actor_only_resume and cfg.policy_only_resume:
      raise ValueError("actor_only_resume and policy_only_resume are mutually exclusive.")
    if cfg.actor_only_resume:
      load_cfg = {"actor": True}
      print("[INFO]: Actor-only resume enabled; optimizer and critic are reinitialized.")
      _load_policy_with_compatible_input_expansion(
        runner,
        resume_path,
        load_cfg=load_cfg,
        map_location=device,
      )
    elif cfg.policy_only_resume:
      load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False}
      print("[INFO]: Policy-only resume enabled; actor and critic are restored, optimizer is reinitialized.")
      _load_policy_with_compatible_input_expansion(
        runner,
        resume_path,
        load_cfg=load_cfg,
        map_location=device,
      )
    else:
      runner.load(str(resume_path), load_cfg=None)
    if cfg.reset_actor_std_on_resume:
      _reset_actor_distribution_std(runner, cfg.agent)

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def _copy_observation_columns(
  old: torch.Tensor,
  target: torch.Tensor,
  projection: _ObservationProjection,
  *,
  fill_value: float,
) -> torch.Tensor | None:
  if old.ndim != 2 or target.ndim != 2:
    return None
  if old.shape[0] != target.shape[0]:
    return None
  if len(projection.stats_source_by_target) != target.shape[1]:
    return None

  resized = target.detach().clone()
  for target_col, source_col in enumerate(projection.stats_source_by_target):
    if source_col is None:
      resized[:, target_col] = fill_value
      continue
    if source_col >= old.shape[1]:
      return None
    resized[:, target_col] = old[:, source_col]
  return resized


def _copy_first_linear_columns(
  old: torch.Tensor,
  target: torch.Tensor,
  projection: _ObservationProjection,
) -> torch.Tensor | None:
  if old.ndim != 2 or target.ndim != 2:
    return None
  if old.shape[0] != target.shape[0]:
    return None
  if len(projection.weight_source_by_target) != target.shape[1]:
    return None

  resized = target.detach().clone()
  resized.zero_()
  for target_col, source_col in enumerate(projection.weight_source_by_target):
    if source_col is None:
      continue
    if source_col >= old.shape[1]:
      return None
    resized[:, target_col] = old[:, source_col]
  return resized


def _expand_observation_normalizer_tensor(
  old: torch.Tensor,
  target: torch.Tensor,
  *,
  fill_value: float,
  projection: _ObservationProjection | None = None,
) -> torch.Tensor | None:
  if old.shape == target.shape:
    return old
  if projection is not None:
    return _copy_observation_columns(
      old,
      target,
      projection,
      fill_value=fill_value,
    )
  if old.ndim != 2 or target.ndim != 2:
    return None
  if old.shape[0] != target.shape[0]:
    return None

  resized = target.detach().clone()
  copy_cols = min(old.shape[1], target.shape[1])
  resized[:, :copy_cols] = old[:, :copy_cols]
  if old.shape[1] < target.shape[1]:
    resized[:, old.shape[1] :] = fill_value
  return resized


def _expand_first_linear_weight(
  old: torch.Tensor,
  target: torch.Tensor,
  *,
  projection: _ObservationProjection | None = None,
) -> torch.Tensor | None:
  if old.shape == target.shape:
    return old
  if projection is not None:
    return _copy_first_linear_columns(old, target, projection)
  if old.ndim != 2 or target.ndim != 2:
    return None
  if old.shape[0] != target.shape[0]:
    return None

  resized = target.detach().clone()
  resized.zero_()
  copy_cols = min(old.shape[1], target.shape[1])
  resized[:, :copy_cols] = old[:, :copy_cols]
  return resized


def _infer_g1_target_action_names(target_rows: int) -> tuple[str, ...] | None:
  if target_rows == len(_G1_23DOF_ACTION_JOINT_NAMES):
    return _G1_23DOF_ACTION_JOINT_NAMES
  if target_rows == len(_G1_ACTION_JOINT_NAMES):
    return _G1_ACTION_JOINT_NAMES
  return None


def _scalar_features(prefix: str, width: int) -> tuple[str, ...]:
  return tuple(f"{prefix}/{idx}" for idx in range(width))


def _joint_features(prefix: str, joint_names: Sequence[str]) -> tuple[str, ...]:
  return tuple(f"{prefix}/{name}" for name in joint_names)


def _bfm_body_state_features(body_names: Sequence[str]) -> tuple[str, ...]:
  features: list[str] = ["bfm/root_height"]
  for name in body_names[1:]:
    for axis in ("x", "y", "z"):
      features.append(f"bfm/body_pos/{name}/{axis}")
  for name in body_names:
    for axis in ("tan_x", "tan_y", "tan_z", "norm_x", "norm_y", "norm_z"):
      features.append(f"bfm/body_rot/{name}/{axis}")
  for name in body_names:
    for axis in ("x", "y", "z"):
      features.append(f"bfm/body_lin_vel/{name}/{axis}")
  for name in body_names:
    for axis in ("x", "y", "z"):
      features.append(f"bfm/body_ang_vel/{name}/{axis}")
  return tuple(features)


def _g1_getup_actor_layout() -> tuple[_ObsTermLayout, ...]:
  return (
    _ObsTermLayout("base_ang_vel", _scalar_features("base_ang_vel", 3), 6),
    _ObsTermLayout("projected_gravity", _scalar_features("projected_gravity", 3), 6),
    _ObsTermLayout("command", _scalar_features("command", 3), 6),
    _ObsTermLayout("joint_pos", _joint_features("joint_pos", _G1_23DOF_ACTION_JOINT_NAMES), 6),
    _ObsTermLayout("joint_vel", _joint_features("joint_vel", _G1_23DOF_ACTION_JOINT_NAMES), 6),
    _ObsTermLayout("actions", _joint_features("actions", _G1_23DOF_ACTION_JOINT_NAMES), 6),
    _ObsTermLayout("getup_progress", _scalar_features("getup_progress", 5), 6),
    _ObsTermLayout("bfm_local_body_state", _bfm_body_state_features(_G1_GETUP_23DOF_BODY_NAMES), 1),
    _ObsTermLayout("height_scan", _scalar_features("height_scan", 187), 6),
  )


def _g1_antifall_getup_actor_layout() -> tuple[_ObsTermLayout, ...]:
  return (
    _ObsTermLayout("base_ang_vel", _scalar_features("base_ang_vel", 3), 3),
    _ObsTermLayout("projected_gravity", _scalar_features("projected_gravity", 3), 3),
    _ObsTermLayout("command", _scalar_features("command", 3), 3),
    _ObsTermLayout("joint_pos", _joint_features("joint_pos", _G1_ACTION_JOINT_NAMES), 3),
    _ObsTermLayout("joint_vel", _joint_features("joint_vel", _G1_ACTION_JOINT_NAMES), 3),
    _ObsTermLayout("actions", _joint_features("actions", _G1_ACTION_JOINT_NAMES), 3),
    _ObsTermLayout("getup_progress", _scalar_features("getup_progress", 5), 3),
    _ObsTermLayout("bfm_local_body_state", _bfm_body_state_features(_G1_ANTIFALL_29DOF_BODY_NAMES), 3),
  )


def _layout_width(layout: Sequence[_ObsTermLayout]) -> int:
  return sum(term.width for term in layout)


def _flatten_layout_indices(
  layout: Sequence[_ObsTermLayout],
) -> tuple[dict[tuple[str, str, int], int], dict[tuple[str, str, int], int]]:
  stats_indices: dict[tuple[str, str, int], int] = {}
  weight_indices: dict[tuple[str, str, int], int] = {}
  offset = 0
  for term in layout:
    for history_idx in range(term.history):
      for feature_idx, feature_name in enumerate(term.feature_names):
        flat_idx = offset + history_idx * len(term.feature_names) + feature_idx
        stats_indices[(term.name, feature_name, history_idx)] = flat_idx
        weight_indices[(term.name, feature_name, history_idx)] = flat_idx
    offset += term.width
  return stats_indices, weight_indices


def _build_observation_projection(
  source_layout: Sequence[_ObsTermLayout],
  target_layout: Sequence[_ObsTermLayout],
) -> _ObservationProjection:
  source_stats, source_weights = _flatten_layout_indices(source_layout)
  stats_cols: list[int | None] = []
  weight_cols: list[int | None] = []
  for target_term in target_layout:
    source_term = next((term for term in source_layout if term.name == target_term.name), None)
    for target_history in range(target_term.history):
      for feature_name in target_term.feature_names:
        if source_term is None:
          stats_cols.append(None)
          weight_cols.append(None)
          continue

        if source_term.history >= target_term.history:
          source_history_idx = target_history + source_term.history - target_term.history
          key = (target_term.name, feature_name, source_history_idx)
          stats_cols.append(source_stats.get(key))
          weight_cols.append(source_weights.get(key))
          continue

        stats_key = (
          target_term.name,
          feature_name,
          min(target_history, source_term.history - 1),
        )
        stats_cols.append(source_stats.get(stats_key))
        if target_history < target_term.history - source_term.history:
          weight_cols.append(None)
          continue
        weight_key = (
          target_term.name,
          feature_name,
          target_history - (target_term.history - source_term.history),
        )
        weight_cols.append(source_weights.get(weight_key))
  return _ObservationProjection(
    stats_source_by_target=tuple(stats_cols),
    weight_source_by_target=tuple(weight_cols),
  )


def _infer_observation_projection(
  old_cols: int,
  target_cols: int,
) -> _ObservationProjection | None:
  getup_actor = _g1_getup_actor_layout()
  antifall_actor = _g1_antifall_getup_actor_layout()
  if old_cols == _layout_width(getup_actor) and target_cols == _layout_width(antifall_actor):
    return _build_observation_projection(getup_actor, antifall_actor)
  return None


def _expand_action_vector_by_name(
  old: torch.Tensor,
  target: torch.Tensor,
  *,
  source_action_names: Sequence[str],
  target_action_names: Sequence[str],
  fill_new_from_target: bool,
) -> torch.Tensor | None:
  if old.shape == target.shape:
    return old
  if old.ndim != 1 or target.ndim != 1:
    return None
  if old.shape[0] != len(source_action_names) or target.shape[0] != len(target_action_names):
    return None

  expanded = target.detach().clone() if fill_new_from_target else torch.zeros_like(target)
  source_by_name = {str(name): idx for idx, name in enumerate(source_action_names)}
  copied = False
  for target_idx, name in enumerate(target_action_names):
    source_idx = source_by_name.get(str(name))
    if source_idx is None:
      continue
    expanded[target_idx] = old[source_idx]
    copied = True
  return expanded if copied else None


def _expand_output_head_by_name(
  old: torch.Tensor,
  target: torch.Tensor,
  *,
  source_action_names: Sequence[str],
  target_action_names: Sequence[str],
) -> torch.Tensor | None:
  if old.shape == target.shape:
    return old
  if old.ndim != 2 or target.ndim != 2:
    return None
  if old.shape[1] != target.shape[1]:
    return None
  if old.shape[0] != len(source_action_names) or target.shape[0] != len(target_action_names):
    return None

  expanded = torch.zeros_like(target)
  source_by_name = {str(name): idx for idx, name in enumerate(source_action_names)}
  copied = False
  for target_idx, name in enumerate(target_action_names):
    source_idx = source_by_name.get(str(name))
    if source_idx is None:
      continue
    expanded[target_idx] = old[source_idx]
    copied = True
  return expanded if copied else None


def _expand_action_output_state(
  checkpoint_state: dict[str, torch.Tensor],
  target_state: dict[str, torch.Tensor],
  *,
  source_action_names: Sequence[str] | None = None,
  target_action_names: Sequence[str] | None = None,
) -> bool:
  old_head = checkpoint_state.get("mlp.6.weight")
  target_head = target_state.get("mlp.6.weight")
  if old_head is None or target_head is None or old_head.shape == target_head.shape:
    return False

  source_names = (
    tuple(source_action_names)
    if source_action_names is not None
    else _infer_g1_target_action_names(int(old_head.shape[0]))
  )
  target_names = (
    tuple(target_action_names)
    if target_action_names is not None
    else _infer_g1_target_action_names(int(target_head.shape[0]))
  )
  if source_names is None or target_names is None:
    return False

  expanded_head = _expand_output_head_by_name(
    old_head,
    target_head,
    source_action_names=source_names,
    target_action_names=target_names,
  )
  if expanded_head is None:
    return False
  checkpoint_state["mlp.6.weight"] = expanded_head

  old_bias = checkpoint_state.get("mlp.6.bias")
  target_bias = target_state.get("mlp.6.bias")
  if old_bias is not None and target_bias is not None:
    expanded_bias = _expand_action_vector_by_name(
      old_bias,
      target_bias,
      source_action_names=source_names,
      target_action_names=target_names,
      fill_new_from_target=False,
    )
    if expanded_bias is None:
      return False
    checkpoint_state["mlp.6.bias"] = expanded_bias

  old_std = checkpoint_state.get("distribution.std_param")
  target_std = target_state.get("distribution.std_param")
  if old_std is not None and target_std is not None:
    expanded_std = _expand_action_vector_by_name(
      old_std,
      target_std,
      source_action_names=source_names,
      target_action_names=target_names,
      fill_new_from_target=True,
    )
    if expanded_std is None:
      return False
    checkpoint_state["distribution.std_param"] = expanded_std

  return True


def _expand_model_input_state(
  checkpoint_state: dict[str, torch.Tensor],
  target_state: dict[str, torch.Tensor],
  *,
  source_action_names: Sequence[str] | None = None,
  target_action_names: Sequence[str] | None = None,
) -> bool:
  """Resize legacy actor/critic tensors for compatible observation dims.

  BFM-local-body observations can add input columns after the existing GetUp
  actor terms, while bridging a richer GetUp checkpoint into an AntiFall task
  can also remove trailing task-local columns.  Leading columns preserve the
  old policy contract.  Newly-added first-layer columns are zeroed, and newly
  added observation normalizer statistics use neutral mean=0,var/std=1 values.
  """

  changed = False
  old_weight = checkpoint_state.get("mlp.0.weight")
  target_weight = target_state.get("mlp.0.weight")
  projection = None
  if old_weight is not None and target_weight is not None and old_weight.ndim == 2 and target_weight.ndim == 2:
    projection = _infer_observation_projection(
      int(old_weight.shape[1]),
      int(target_weight.shape[1]),
    )

  normalizer_fill = {
    "obs_normalizer._mean": 0.0,
    "obs_normalizer._var": 1.0,
    "obs_normalizer._std": 1.0,
  }
  for key, fill_value in normalizer_fill.items():
    old = checkpoint_state.get(key)
    target = target_state.get(key)
    if old is None or target is None:
      continue
    expanded = _expand_observation_normalizer_tensor(
      old,
      target,
      fill_value=fill_value,
      projection=projection,
    )
    if expanded is None:
      continue
    if expanded.shape != old.shape:
      changed = True
    checkpoint_state[key] = expanded

  if old_weight is not None and target_weight is not None:
    expanded_weight = _expand_first_linear_weight(
      old_weight,
      target_weight,
      projection=projection,
    )
    if expanded_weight is not None:
      if expanded_weight.shape != old_weight.shape:
        changed = True
      checkpoint_state["mlp.0.weight"] = expanded_weight

  if _expand_action_output_state(
    checkpoint_state,
    target_state,
    source_action_names=source_action_names,
    target_action_names=target_action_names,
  ):
    changed = True

  return changed


def _load_policy_with_compatible_input_expansion(
  runner,
  resume_path: Path,
  *,
  load_cfg: dict | None,
  map_location: str | None,
) -> None:
  """Load actor/critic checkpoints, expanding input tensors for additive obs terms."""

  try:
    try:
      runner.load(str(resume_path), load_cfg=load_cfg, map_location=map_location)
    except TypeError:
      runner.load(str(resume_path), load_cfg=load_cfg)
    return
  except RuntimeError as exc:
    if "size mismatch" not in str(exc):
      raise
    checkpoint = torch.load(resume_path, map_location=map_location, weights_only=False)
    changed_parts: list[str] = []

    if load_cfg.get("actor"):
      actor_state = checkpoint.get("actor_state_dict")
      if not isinstance(actor_state, dict):
        raise
      target_actor_state = runner.alg.get_policy().state_dict()
      if _expand_model_input_state(actor_state, target_actor_state):
        changed_parts.append("actor")
      runner.alg.get_policy().load_state_dict(actor_state, strict=True)

    if load_cfg.get("critic"):
      critic_state = checkpoint.get("critic_state_dict")
      if not isinstance(critic_state, dict):
        raise
      target_critic_state = runner.alg.critic.state_dict()
      if _expand_model_input_state(critic_state, target_critic_state):
        changed_parts.append("critic")
      runner.alg.critic.load_state_dict(critic_state, strict=True)

    if not changed_parts:
      raise

    print(
      "[INFO]: Resized "
      + "/".join(changed_parts)
      + " checkpoint input tensors for compatible observation dimensions; "
      + "optimizer state was not restored."
    )


# Backward-compatible name used by rollout diagnostics that only request actor load.
_load_actor_with_compatible_input_expansion = _load_policy_with_compatible_input_expansion


def _reset_actor_distribution_std(runner, agent_cfg: RslRlBaseRunnerCfg) -> None:
  """Reset loaded Gaussian actor std to the current config's initial std."""

  distribution_cfg = getattr(getattr(agent_cfg, "actor", None), "distribution_cfg", {}) or {}
  init_std = float(distribution_cfg.get("init_std", 1.0))
  std_type = distribution_cfg.get("std_type", "scalar")
  policy = runner.alg.get_policy()
  distribution = getattr(policy, "distribution", None)
  if distribution is None:
    raise RuntimeError("Cannot reset actor std: policy has no distribution module.")
  if std_type == "scalar" and hasattr(distribution, "std_param"):
    distribution.std_param.data.fill_(init_std)
    print(f"[INFO]: Reset actor std_param to {init_std:g}.")
    return
  if std_type == "log" and hasattr(distribution, "log_std_param"):
    import math

    distribution.log_std_param.data.fill_(math.log(init_std))
    print(f"[INFO]: Reset actor log_std_param to log({init_std:g}).")
    return
  raise RuntimeError(
    f"Cannot reset actor std: unsupported distribution std_type={std_type!r}."
  )


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)
  if task_id == "Unitree-G1-GetUp" and args.getup_terrain is not None:
    from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg
    from src.tasks.velocity.config.g1_getup.rl_cfg import unitree_g1_getup_ppo_runner_cfg

    terrain_env_cfg = unitree_g1_getup_env_cfg(terrain=args.getup_terrain)
    terrain_agent_cfg = unitree_g1_getup_ppo_runner_cfg(terrain=args.getup_terrain)
    terrain_env_cfg.scene.num_envs = args.env.scene.num_envs
    terrain_agent_cfg.max_iterations = args.agent.max_iterations
    terrain_agent_cfg.num_steps_per_env = args.agent.num_steps_per_env
    terrain_agent_cfg.save_interval = args.agent.save_interval
    terrain_agent_cfg.obs_groups = args.agent.obs_groups
    terrain_agent_cfg.logger = args.agent.logger
    terrain_agent_cfg.upload_model = args.agent.upload_model
    terrain_agent_cfg.seed = args.agent.seed
    terrain_agent_cfg.resume = args.agent.resume
    terrain_agent_cfg.load_run = args.agent.load_run
    terrain_agent_cfg.load_checkpoint = args.agent.load_checkpoint
    terrain_agent_cfg.clip_actions = args.agent.clip_actions
    terrain_agent_cfg.actor = args.agent.actor
    terrain_agent_cfg.critic = args.agent.critic
    terrain_agent_cfg.algorithm = args.agent.algorithm
    args = replace(args, env=terrain_env_cfg, agent=terrain_agent_cfg)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(_parse_gpu_ids_arg(args.gpu_ids))

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=_normalize_gpu_ids_cli_args(remaining_args),
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
