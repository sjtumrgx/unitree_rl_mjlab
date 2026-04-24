"""Runtime helpers for ``scripts/play_parkour.py``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping

import numpy as np
import torch

from src.parkour.contract import (
  ACTION_SIZE,
  DEPTH_HEIGHT,
  DEPTH_HISTORY,
  DEPTH_SIZE,
  DEPTH_WIDTH,
  ONNX_POLICY_JOINT_NAMES,
  POLICY_HISTORY,
  PROPRIO_GROUP_DIMS,
  PROPRIO_SIZE,
  ParkourDeployContract,
  constant_depth_stack,
  vector_stats,
)


def align_policy_body_vector(values: np.ndarray) -> np.ndarray:
  """Apply the C++ ParkourArticulation -90deg-Y policy-frame transform."""
  arr = np.asarray(values, dtype=np.float32)
  return np.stack((-arr[..., 2], arr[..., 1], arr[..., 0]), axis=-1)


def _tensor_env0(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()[0].astype(np.float32, copy=False)


@dataclass(frozen=True)
class ParkourFrameDiagnostics:
  frame_mode: str
  raw_base_ang_vel: list[float]
  policy_base_ang_vel: list[float]
  raw_projected_gravity: list[float]
  policy_projected_gravity: list[float]
  command: list[float]
  joint_pos_rel_stats: Mapping[str, float]
  joint_vel_rel_stats: Mapping[str, float]


@dataclass(frozen=True)
class ParkourMappingProof:
  joint_order: str
  action_order: str
  onnx_policy_joint_names_head: list[str]
  onnx_policy_joint_names_tail: list[str]
  policy_joint_names_head: list[str]
  policy_joint_names_tail: list[str]
  robot_joint_names_head: list[str]
  robot_joint_names_tail: list[str]
  env_action_target_names_head: list[str]
  env_action_target_names_tail: list[str]
  policy_to_robot_joint_indices: list[int]
  env_action_to_policy_indices: list[int]

  def as_dict(self) -> dict[str, Any]:
    return {
      "policy_joint_names_head": self.policy_joint_names_head,
      "joint_order": self.joint_order,
      "action_order": self.action_order,
      "onnx_policy_joint_names_head": self.onnx_policy_joint_names_head,
      "onnx_policy_joint_names_tail": self.onnx_policy_joint_names_tail,
      "policy_joint_names_tail": self.policy_joint_names_tail,
      "robot_joint_names_head": self.robot_joint_names_head,
      "robot_joint_names_tail": self.robot_joint_names_tail,
      "env_action_target_names_head": self.env_action_target_names_head,
      "env_action_target_names_tail": self.env_action_target_names_tail,
      "policy_to_robot_joint_indices": self.policy_to_robot_joint_indices,
      "env_action_to_policy_indices": self.env_action_to_policy_indices,
    }


class ConstantDepthProvider:
  def __init__(self, value: float) -> None:
    self._stack = constant_depth_stack(value)

  def stack(self, adapter: "ParkourObservationAdapter | None" = None) -> np.ndarray:
    return self._stack.copy()

  def diagnostics(self) -> dict[str, Any]:
    return {
      "mode": "constant",
      "shape": list(self._stack.shape),
      "size": int(self._stack.size),
      "stats": vector_stats(self._stack),
    }


class FlatGroundDepthProvider:
  """Analytic flat-floor depth for the training ray-caster camera contract.

  This is intentionally lighter than full MuJoCo camera rendering: it preserves
  the exported policy's depth shape, crop, FOV, normalization, and head-camera
  pose while avoiding renderer/camera parity as a first-stage blocker.
  """

  _RAW_WIDTH = 64
  _RAW_HEIGHT = 36
  _CROP_TOP = 18
  _CROP_BOTTOM = 0
  _CROP_LEFT = 16
  _CROP_RIGHT = 16
  _MAX_DEPTH_M = 2.5
  _MIN_DEPTH_M = 0.0
  _FOV_X_DEG = 89.51
  _FOV_Y_DEG = 58.29
  _CAMERA_POS_B = np.asarray(
    [0.0487988662332928, 0.01, 0.4378029937970051],
    dtype=np.float32,
  )
  # Training NoisyGroupedRayCasterCameraCfg offset, wxyz, convention="world".
  _CAMERA_QUAT_B = np.asarray(
    [0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0],
    dtype=np.float32,
  )

  def __init__(self) -> None:
    self._last_stack = constant_depth_stack(1.0)

  @staticmethod
  def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1.0e-8)
    w, x, y, z = q.tolist()
    return np.asarray(
      [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
      ],
      dtype=np.float32,
    )

  @staticmethod
  def _yaw_matrix_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float32).tolist()
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

  def _render_frame(self, adapter: "ParkourObservationAdapter") -> np.ndarray:
    root_pos = _tensor_env0(adapter.robot.data.root_link_pos_w)
    root_quat = _tensor_env0(adapter.robot.data.root_link_quat_w)
    yaw_rot = self._yaw_matrix_from_quat_wxyz(root_quat)
    camera_pos = root_pos + yaw_rot @ self._CAMERA_POS_B
    camera_rot = yaw_rot @ self._quat_wxyz_to_matrix(self._CAMERA_QUAT_B)

    tan_x = np.tan(np.deg2rad(self._FOV_X_DEG) * 0.5)
    tan_y = np.tan(np.deg2rad(self._FOV_Y_DEG) * 0.5)
    cols = (np.arange(self._RAW_WIDTH, dtype=np.float32) + 0.5) / self._RAW_WIDTH
    rows = (np.arange(self._RAW_HEIGHT, dtype=np.float32) + 0.5) / self._RAW_HEIGHT
    # Camera +X is optical forward for the training ray pattern; +Z is image up.
    image_x = (cols * 2.0 - 1.0) * tan_x
    image_z = (1.0 - rows * 2.0) * tan_y
    y_grid, z_grid = np.meshgrid(image_x, image_z)
    dirs_camera = np.stack(
      [np.ones_like(y_grid), y_grid, z_grid],
      axis=-1,
    ).astype(np.float32)
    dirs_world = dirs_camera @ camera_rot.T
    dz = dirs_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
      depth_m = np.where(dz < -1.0e-6, -float(camera_pos[2]) / dz, self._MAX_DEPTH_M)
    depth_m = np.clip(depth_m, self._MIN_DEPTH_M, self._MAX_DEPTH_M)
    normalized = (depth_m - self._MIN_DEPTH_M) / (self._MAX_DEPTH_M - self._MIN_DEPTH_M)
    cropped = normalized[
      self._CROP_TOP : self._RAW_HEIGHT - self._CROP_BOTTOM,
      self._CROP_LEFT : self._RAW_WIDTH - self._CROP_RIGHT,
    ]
    if cropped.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
      raise RuntimeError(f"flat-ground depth frame shape {cropped.shape}; expected {(DEPTH_HEIGHT, DEPTH_WIDTH)}")
    return cropped.astype(np.float32, copy=False)

  def stack(self, adapter: "ParkourObservationAdapter | None" = None) -> np.ndarray:
    if adapter is None:
      return self._last_stack.copy()
    frame = self._render_frame(adapter)
    self._last_stack = np.repeat(frame[None, :, :], DEPTH_HISTORY, axis=0).astype(np.float32, copy=False)
    return self._last_stack.copy()

  def diagnostics(self) -> dict[str, Any]:
    return {
      "mode": "flat-ground",
      "shape": list(self._last_stack.shape),
      "size": int(self._last_stack.size),
      "stats": vector_stats(self._last_stack),
      "camera": {
        "raw_resolution": [self._RAW_WIDTH, self._RAW_HEIGHT],
        "crop_region": [self._CROP_TOP, self._CROP_BOTTOM, self._CROP_LEFT, self._CROP_RIGHT],
        "output_resolution": [DEPTH_WIDTH, DEPTH_HEIGHT],
      },
    }


class ParkourObservationAdapter:
  """Build parkour proprio history from raw MJLab/MuJoCo state."""

  def __init__(
    self,
    env: Any,
    contract: ParkourDeployContract,
    *,
    command: tuple[float, float, float],
    frame_mode: str = "mjlab",
    joint_order: str = "isaac",
    action_order: str = "isaac",
  ) -> None:
    self.env = env
    self.contract = contract
    self.command = np.asarray(command, dtype=np.float32)
    if frame_mode not in {"mjlab", "deploy-align"}:
      raise ValueError("frame_mode must be 'mjlab' or 'deploy-align'")
    if joint_order not in {"isaac", "policy", "robot"}:
      raise ValueError("joint_order must be 'isaac', 'policy', or 'robot'")
    if action_order not in {"isaac", "policy", "env"}:
      raise ValueError("action_order must be 'isaac', 'policy', or 'env'")
    self.frame_mode = frame_mode
    self.joint_order = joint_order
    self.action_order = action_order
    self.robot = env.scene["robot"]
    self.deploy_joint_names = tuple(contract.joint_names)
    self.onnx_policy_joint_names = tuple(ONNX_POLICY_JOINT_NAMES)
    robot_joint_names = list(self.robot.joint_names)
    name_to_robot_index = {name: idx for idx, name in enumerate(robot_joint_names)}
    required_joint_names = dict.fromkeys((*self.deploy_joint_names, *self.onnx_policy_joint_names))
    missing = [
      name
      for name in required_joint_names
      if name not in name_to_robot_index
    ]
    if missing:
      raise RuntimeError(f"Robot is missing parkour policy joints: {missing}")
    if self.joint_order == "isaac":
      self.observation_joint_names = self.onnx_policy_joint_names
      self.policy_to_robot_joint_indices = [
        name_to_robot_index[name] for name in self.onnx_policy_joint_names
      ]
    elif self.joint_order == "policy":
      self.observation_joint_names = self.deploy_joint_names
      self.policy_to_robot_joint_indices = [name_to_robot_index[name] for name in self.deploy_joint_names]
    else:
      self.observation_joint_names = tuple(robot_joint_names)
      self.policy_to_robot_joint_indices = list(range(len(robot_joint_names)))

    self.env_action_target_names = self._collect_env_action_target_names()
    if self.action_order == "isaac":
      action_joint_names = self.onnx_policy_joint_names
    elif self.action_order == "policy":
      action_joint_names = self.deploy_joint_names
    else:
      action_joint_names = tuple(self.env_action_target_names)
    name_to_policy_index = {name: idx for idx, name in enumerate(action_joint_names)}
    if self.action_order in {"isaac", "policy"}:
      missing_action_targets = [
        name for name in self.env_action_target_names if name not in name_to_policy_index
      ]
      if missing_action_targets:
        raise RuntimeError(
          "Action manager controls joints outside the parkour policy contract: "
          f"{missing_action_targets}"
        )
      self.env_action_to_policy_indices = [
        name_to_policy_index[name] for name in self.env_action_target_names
      ]
    else:
      self.env_action_to_policy_indices = list(range(len(self.env_action_target_names)))
    if len(self.env_action_to_policy_indices) != ACTION_SIZE:
      raise RuntimeError(
        f"Action manager dimension is {len(self.env_action_to_policy_indices)}, expected {ACTION_SIZE}"
      )

    self._history: dict[str, Deque[np.ndarray]] = {
      name: deque(maxlen=POLICY_HISTORY) for name in PROPRIO_GROUP_DIMS
    }

  def _collect_env_action_target_names(self) -> list[str]:
    names: list[str] = []
    for term_name in self.env.action_manager.active_terms:
      term = self.env.action_manager.get_term(term_name)
      target_names = getattr(term, "target_names", None)
      if target_names is None:
        raise RuntimeError(f"Action term {term_name!r} does not expose target_names")
      names.extend(list(target_names))
    return names

  def mapping_proof(self) -> ParkourMappingProof:
    robot_joint_names = list(self.robot.joint_names)
    return ParkourMappingProof(
      joint_order=self.joint_order,
      action_order=self.action_order,
      onnx_policy_joint_names_head=list(self.onnx_policy_joint_names[:6]),
      onnx_policy_joint_names_tail=list(self.onnx_policy_joint_names[-6:]),
      policy_joint_names_head=list(self.observation_joint_names[:6]),
      policy_joint_names_tail=list(self.observation_joint_names[-6:]),
      robot_joint_names_head=robot_joint_names[:6],
      robot_joint_names_tail=robot_joint_names[-6:],
      env_action_target_names_head=self.env_action_target_names[:6],
      env_action_target_names_tail=self.env_action_target_names[-6:],
      policy_to_robot_joint_indices=self.policy_to_robot_joint_indices,
      env_action_to_policy_indices=self.env_action_to_policy_indices,
    )

  def set_fixed_command(self) -> None:
    term = self.env.command_manager.get_term("twist")
    command = torch.tensor(self.command, dtype=torch.float32, device=self.env.device)
    term.command[:] = command.unsqueeze(0).repeat(self.env.num_envs, 1)

  def _sensor_or_root_ang_vel(self) -> np.ndarray:
    sensors = getattr(self.env.scene, "sensors", {})
    if "robot/imu_gyro" in sensors:
      return _tensor_env0(self.env.scene["robot/imu_gyro"].data)
    return _tensor_env0(self.robot.data.root_link_ang_vel_b)

  def latest_terms(self, *, last_policy_action: np.ndarray) -> tuple[dict[str, np.ndarray], ParkourFrameDiagnostics]:
    idx = torch.tensor(self.policy_to_robot_joint_indices, device=self.env.device, dtype=torch.long)
    raw_base_ang_vel = self._sensor_or_root_ang_vel()
    raw_projected_gravity = _tensor_env0(self.robot.data.projected_gravity_b)
    if self.frame_mode == "deploy-align":
      policy_base_ang_vel = align_policy_body_vector(raw_base_ang_vel)
      policy_projected_gravity = align_policy_body_vector(raw_projected_gravity)
    else:
      policy_base_ang_vel = raw_base_ang_vel
      policy_projected_gravity = raw_projected_gravity

    joint_pos = _tensor_env0(self.robot.data.joint_pos[:, idx])
    joint_vel = _tensor_env0(self.robot.data.joint_vel[:, idx])
    default_joint_pos = _tensor_env0(self.robot.data.default_joint_pos[:, idx])
    default_joint_vel = _tensor_env0(self.robot.data.default_joint_vel[:, idx])
    joint_pos_rel = joint_pos - default_joint_pos
    joint_vel_rel = joint_vel - default_joint_vel
    last_action = np.asarray(last_policy_action, dtype=np.float32).reshape(ACTION_SIZE)

    terms = {
      "base_ang_vel": self.contract.scaled_group("base_ang_vel", policy_base_ang_vel),
      "projected_gravity": self.contract.scaled_group("projected_gravity", policy_projected_gravity),
      "velocity_commands": self.contract.scaled_group("velocity_commands", self.command),
      "joint_pos_rel": self.contract.scaled_group("joint_pos_rel", joint_pos_rel),
      "joint_vel_rel": self.contract.scaled_group("joint_vel_rel", joint_vel_rel),
      "last_action": self.contract.scaled_group("last_action", last_action),
    }
    diagnostics = ParkourFrameDiagnostics(
      frame_mode=self.frame_mode,
      raw_base_ang_vel=raw_base_ang_vel.tolist(),
      policy_base_ang_vel=policy_base_ang_vel.tolist(),
      raw_projected_gravity=raw_projected_gravity.tolist(),
      policy_projected_gravity=policy_projected_gravity.tolist(),
      command=self.command.tolist(),
      joint_pos_rel_stats=vector_stats(joint_pos_rel),
      joint_vel_rel_stats=vector_stats(joint_vel_rel),
    )
    return terms, diagnostics

  def warm_start(self, *, last_policy_action: np.ndarray | None = None) -> ParkourFrameDiagnostics:
    action = np.zeros(ACTION_SIZE, dtype=np.float32) if last_policy_action is None else last_policy_action
    terms, diagnostics = self.latest_terms(last_policy_action=action)
    for name, values in terms.items():
      self._history[name].clear()
      for _ in range(POLICY_HISTORY):
        self._history[name].append(values.copy())
    return diagnostics

  def append_current(self, *, last_policy_action: np.ndarray) -> ParkourFrameDiagnostics:
    terms, diagnostics = self.latest_terms(last_policy_action=last_policy_action)
    for name, values in terms.items():
      self._history[name].append(values.copy())
    return diagnostics

  def proprio(self) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for group_name in PROPRIO_GROUP_DIMS:
      history = self._history[group_name]
      if len(history) != POLICY_HISTORY:
        raise RuntimeError(
          f"History for {group_name} has {len(history)} frames; expected {POLICY_HISTORY}"
        )
      pieces.extend(history)
    proprio = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    if proprio.shape != (PROPRIO_SIZE,):
      raise RuntimeError(f"Proprio vector shape {proprio.shape}, expected {(PROPRIO_SIZE,)}")
    return proprio

  def env_action_from_policy_action(self, policy_action: np.ndarray) -> np.ndarray:
    action = np.asarray(policy_action, dtype=np.float32).reshape(ACTION_SIZE)
    return action[self.env_action_to_policy_indices].reshape(1, -1)

  def fall_signals(self) -> dict[str, Any]:
    raw_projected_gravity = _tensor_env0(self.robot.data.projected_gravity_b)
    root_pos = _tensor_env0(self.robot.data.root_link_pos_w)
    root_quat = _tensor_env0(self.robot.data.root_link_quat_w)
    return {
      "base_pos": root_pos.tolist(),
      "base_height": float(root_pos[2]),
      "root_quat": root_quat.tolist(),
      "raw_projected_gravity": raw_projected_gravity.tolist(),
      "raw_projected_gravity_z": float(raw_projected_gravity[2]),
    }


def classify_failure(
  *,
  reason: str,
  fall_signals: Mapping[str, Any] | None = None,
  reset_count: int = 0,
  onnx_ok: bool = True,
  action_finite: bool = True,
  proprio_finite: bool = True,
  depth_mode: str = "constant",
) -> str:
  if not onnx_ok:
    return "ONNX shape mismatch"
  if not proprio_finite:
    return "observation ordering/history mismatch"
  if not action_finite:
    return "joint/action mapping mismatch"
  if reset_count > 0:
    return "termination/contact issue"
  if fall_signals:
    height = float(fall_signals.get("base_height", 1.0))
    gravity_z = float(fall_signals.get("raw_projected_gravity_z", -1.0))
    if reason == "zero_action_fall":
      return "asset/init pose mismatch"
    if height < 0.45:
      return "asset/init pose mismatch" if reason == "initial_fall" else "joint/action mapping mismatch"
    if gravity_z > -0.25:
      return "observation frame mismatch"
  if depth_mode != "constant":
    return "depth input issue"
  return "unknown; inspect diagnostics for observation frame/action mapping"


def make_depth_provider(depth_mode: str, constant_depth: float, env: Any | None = None) -> ConstantDepthProvider | FlatGroundDepthProvider:
  if depth_mode == "constant":
    return ConstantDepthProvider(constant_depth)
  if depth_mode == "flat-ground":
    return FlatGroundDepthProvider()
  raise NotImplementedError(
    "Full MuJoCo depth rendering is intentionally gated as stage 2 renderer parity work. "
    "Use --depth-mode constant to isolate proprio/action/asset alignment or "
    "--depth-mode flat-ground for an analytic flat-floor depth ablation."
  )


def assert_depth_contract(depth_stack: np.ndarray) -> None:
  if depth_stack.size != DEPTH_SIZE:
    raise RuntimeError(f"Depth stack size {depth_stack.size}; expected {DEPTH_SIZE}")
  if not np.isfinite(depth_stack).all():
    raise RuntimeError("Depth stack contains NaN/Inf")
