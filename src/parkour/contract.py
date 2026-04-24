"""Deploy-contract helpers for the G1 parkour MuJoCo debug runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = REPO_ROOT / "deploy" / "robots" / "g1_parkour" / "config" / "policy" / "parkour" / "v0"
DEFAULT_EXPORTED_DIR = DEFAULT_POLICY_DIR / "exported"
DEFAULT_DEPLOY_YAML = DEFAULT_POLICY_DIR / "params" / "deploy.yaml"

DEPTH_HISTORY = 8
DEPTH_HEIGHT = 18
DEPTH_WIDTH = 32
DEPTH_SHAPE = (DEPTH_HISTORY, DEPTH_HEIGHT, DEPTH_WIDTH)
DEPTH_ENCODER_INPUT_SHAPE = (1, DEPTH_HISTORY, DEPTH_HEIGHT, DEPTH_WIDTH)
DEPTH_ENCODER_OUTPUT_SHAPE = (1, 128)
DEPTH_SIZE = DEPTH_HISTORY * DEPTH_HEIGHT * DEPTH_WIDTH
DEPTH_LATENT_SIZE = 128
POLICY_HISTORY = 8
ACTION_SIZE = 29
PROPRIO_SIZE = 768
ACTOR_INPUT_SIZE = PROPRIO_SIZE + DEPTH_LATENT_SIZE
ACTOR_INPUT_SHAPE = (1, ACTOR_INPUT_SIZE)
ACTOR_OUTPUT_SHAPE = (1, ACTION_SIZE)

TRAINING_JOINT_NAMES: tuple[str, ...] = (
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

# IsaacLab's exported ONNX actor keeps the action/proprio order produced by
# the training articulation/action manager.  The deploy YAML above is
# motor-oriented (legs, waist, arms) for the C++ lowstate path; the ONNX model
# itself was exported before that deploy-order promotion step.  Keep both
# orders explicit so the MuJoCo debug harness can feed the actor in training
# order while still applying named joint scales/offsets from deploy.yaml.
ONNX_POLICY_JOINT_NAMES: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "waist_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "waist_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "waist_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
)

PROPRIO_GROUP_DIMS: Mapping[str, int] = {
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "velocity_commands": 3,
  "joint_pos_rel": ACTION_SIZE,
  "joint_vel_rel": ACTION_SIZE,
  "last_action": ACTION_SIZE,
}
STALE_PARKOUR_SCENE_SENSOR_NAMES = ("robot/imu_ang_vel", "robot/imu_lin_vel")
PARKOUR_SCENE_SENSOR_REMAP = {
  "robot/imu_ang_vel": "robot/imu_gyro",
  "robot/imu_lin_vel": "robot/frame_vel",
}


@dataclass(frozen=True)
class ParkourModelPaths:
  policy_dir: Path
  exported_dir: Path
  deploy_yaml: Path
  depth_encoder_onnx: Path
  actor_onnx: Path


@dataclass(frozen=True)
class ParkourDeployContract:
  deploy_yaml: Path
  action_scales: tuple[float, ...]
  action_offsets: tuple[float, ...]
  proprio_scales: Mapping[str, tuple[float, ...]]
  proprio_history_lengths: Mapping[str, int]
  depth_shape: tuple[int, int, int] = DEPTH_SHAPE
  depth_size: int = DEPTH_SIZE
  depth_latent_size: int = DEPTH_LATENT_SIZE
  proprio_size: int = PROPRIO_SIZE
  actor_input_size: int = ACTOR_INPUT_SIZE
  action_size: int = ACTION_SIZE
  joint_names: tuple[str, ...] = TRAINING_JOINT_NAMES

  @property
  def action_scale_by_joint(self) -> dict[str, float]:
    return dict(zip(self.joint_names, self.action_scales, strict=True))

  @property
  def action_offset_by_joint(self) -> dict[str, float]:
    return dict(zip(self.joint_names, self.action_offsets, strict=True))

  def scaled_group(self, group_name: str, values: np.ndarray) -> np.ndarray:
    scales = np.asarray(self.proprio_scales[group_name], dtype=np.float32)
    if values.shape[-1] != scales.shape[-1]:
      raise ValueError(
        f"{group_name} has dim {values.shape[-1]}, expected {scales.shape[-1]}"
      )
    return values.astype(np.float32, copy=False) * scales


def _as_float_tuple(value: Any, *, expected_len: int, name: str) -> tuple[float, ...]:
  if isinstance(value, (int, float)):
    values = (float(value),) * expected_len
  else:
    values = tuple(float(item) for item in value)
  if len(values) != expected_len:
    raise ValueError(f"{name} expected {expected_len} values, got {len(values)}")
  return values


def _load_yaml(path: Path) -> dict[str, Any]:
  if not path.exists():
    raise FileNotFoundError(f"Deploy YAML not found: {path}")
  payload = yaml.safe_load(path.read_text())
  if not isinstance(payload, dict):
    raise ValueError(f"Deploy YAML did not contain a mapping: {path}")
  return payload


def resolve_model_paths(
  *,
  policy_dir: Path | str | None = None,
  exported_dir: Path | str | None = None,
) -> ParkourModelPaths:
  resolved_policy_dir = Path(policy_dir or DEFAULT_POLICY_DIR).expanduser().resolve()
  resolved_exported_dir = Path(exported_dir).expanduser().resolve() if exported_dir else resolved_policy_dir / "exported"
  deploy_yaml = resolved_policy_dir / "params" / "deploy.yaml"
  return ParkourModelPaths(
    policy_dir=resolved_policy_dir,
    exported_dir=resolved_exported_dir,
    deploy_yaml=deploy_yaml,
    depth_encoder_onnx=resolved_exported_dir / "0-depth_encoder.onnx",
    actor_onnx=resolved_exported_dir / "actor.onnx",
  )


def load_deploy_contract(deploy_yaml: Path | str = DEFAULT_DEPLOY_YAML) -> ParkourDeployContract:
  path = Path(deploy_yaml).expanduser().resolve()
  payload = _load_yaml(path)
  action_cfg = payload["actions"]["JointPositionAction"]
  action_scales = _as_float_tuple(action_cfg["scale"], expected_len=ACTION_SIZE, name="action scale")
  action_offsets = _as_float_tuple(action_cfg["offset"], expected_len=ACTION_SIZE, name="action offset")

  proprio_cfg = payload["observations"]["proprio"]
  proprio_scales: dict[str, tuple[float, ...]] = {}
  proprio_history_lengths: dict[str, int] = {}
  for group_name, dim in PROPRIO_GROUP_DIMS.items():
    group_cfg = proprio_cfg[group_name]
    proprio_scales[group_name] = _as_float_tuple(
      group_cfg.get("scale", [1.0] * dim), expected_len=dim, name=f"{group_name} scale"
    )
    proprio_history_lengths[group_name] = int(group_cfg["history_length"])

  contract = ParkourDeployContract(
    deploy_yaml=path,
    action_scales=action_scales,
    action_offsets=action_offsets,
    proprio_scales=proprio_scales,
    proprio_history_lengths=proprio_history_lengths,
  )
  validate_deploy_contract(contract)
  return contract


def validate_deploy_contract(contract: ParkourDeployContract) -> None:
  if len(contract.joint_names) != ACTION_SIZE:
    raise ValueError(f"Expected {ACTION_SIZE} joint names, got {len(contract.joint_names)}")
  if len(contract.action_scales) != ACTION_SIZE:
    raise ValueError(f"Expected {ACTION_SIZE} action scales, got {len(contract.action_scales)}")
  if len(contract.action_offsets) != ACTION_SIZE:
    raise ValueError(f"Expected {ACTION_SIZE} action offsets, got {len(contract.action_offsets)}")
  total = 0
  for group_name, dim in PROPRIO_GROUP_DIMS.items():
    history = contract.proprio_history_lengths[group_name]
    if history != POLICY_HISTORY:
      raise ValueError(f"{group_name} history_length={history}; expected {POLICY_HISTORY}")
    total += dim * history
  if total != PROPRIO_SIZE:
    raise ValueError(f"Proprio size math produced {total}; expected {PROPRIO_SIZE}")


def validate_model_files(paths: ParkourModelPaths) -> None:
  missing = [path for path in (paths.deploy_yaml, paths.depth_encoder_onnx, paths.actor_onnx) if not path.exists()]
  if missing:
    missing_text = "\n".join(str(path) for path in missing)
    raise FileNotFoundError(f"Missing parkour policy artifact(s):\n{missing_text}")


def constant_depth_stack(value: float) -> np.ndarray:
  if not np.isfinite(value):
    raise ValueError(f"constant depth must be finite, got {value!r}")
  clipped = float(np.clip(value, 0.0, 1.0))
  return np.full(DEPTH_SHAPE, clipped, dtype=np.float32)


def vector_stats(values: np.ndarray | Iterable[float]) -> dict[str, float]:
  arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32)
  if arr.size == 0:
    return {"min": 0.0, "max": 0.0, "mean": 0.0}
  return {
    "min": float(np.nanmin(arr)),
    "max": float(np.nanmax(arr)),
    "mean": float(np.nanmean(arr)),
  }


def shape_as_tuple(shape: Iterable[Any]) -> tuple[int, ...]:
  parsed: list[int] = []
  for dim in shape:
    if not isinstance(dim, int):
      raise ValueError(f"Dynamic ONNX shape dimension is not supported: {shape}")
    parsed.append(dim)
  return tuple(parsed)


def find_stale_sensor_references(env_cfg: Any) -> list[str]:
  stale: list[str] = []
  for group in getattr(env_cfg, "observations", {}).values():
    for term in getattr(group, "terms", {}).values():
      sensor_name = (getattr(term, "params", {}) or {}).get("sensor_name")
      if sensor_name in STALE_PARKOUR_SCENE_SENSOR_NAMES:
        stale.append(sensor_name)
  return stale


def assert_no_stale_sensor_references(env_cfg: Any) -> None:
  stale = find_stale_sensor_references(env_cfg)
  if stale:
    raise RuntimeError(
      "Parkour scene uses imu_gyro/frame_vel sensors, but stale velocity-env "
      f"sensor reference(s) remain: {sorted(set(stale))}"
    )
