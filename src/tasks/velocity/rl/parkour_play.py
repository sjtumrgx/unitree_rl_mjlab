"""Runtime helpers for ``scripts/play_parkour.py``."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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
  ParkourDepthInterfaceContract,
  constant_depth_stack,
  load_depth_interface_contract,
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

  def latest_frame(self, frame_kind: str = "policy") -> np.ndarray:
    del frame_kind
    return self._stack[-1].copy()

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

  def latest_frame(self, frame_kind: str = "policy") -> np.ndarray:
    del frame_kind
    return self._last_stack[-1].copy()

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


def _write_grayscale_pgm(path: Path, frame: np.ndarray) -> None:
  """Write a normalized 2D depth frame as a binary PGM without extra deps."""
  path.parent.mkdir(parents=True, exist_ok=True)
  image = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
  payload = (image * 255.0).astype(np.uint8)
  path.write_bytes(f"P5\n{payload.shape[1]} {payload.shape[0]}\n255\n".encode() + payload.tobytes())


class MujocoRendererDepthProvider:
  """Offscreen MuJoCo renderer depth provider for the parkour camera contract."""

  def __init__(
    self,
    *,
    env: Any,
    contract: ParkourDepthInterfaceContract | None = None,
    debug_dir: Path | None = None,
  ) -> None:
    self.env = env
    self.contract = contract or load_depth_interface_contract()
    self.debug_dir = debug_dir.expanduser() if debug_dir is not None else None
    self._mujoco: Any | None = None
    self._renderer: Any | None = None
    self._render_data: Any | None = None
    self._camera_name: str | None = None
    self._source_history: Deque[np.ndarray] = deque(maxlen=self.contract.history_source_length)
    self._last_raw_frame = np.ones((self.contract.raw_height, self.contract.raw_width), dtype=np.float32)
    self._last_frame = np.ones((self.contract.output_height, self.contract.output_width), dtype=np.float32)
    self._last_stack = constant_depth_stack(1.0)
    self._last_baseline: np.ndarray | None = None
    self._last_visibility: dict[str, Any] | None = None
    self._best_visibility: dict[str, Any] | None = None
    self._visibility_samples = 0
    self._preview_paths: dict[str, str] = {}
    self._flat_baseline = FlatGroundDepthProvider()

  def reset(self) -> None:
    self._source_history.clear()
    self._last_raw_frame = np.ones((self.contract.raw_height, self.contract.raw_width), dtype=np.float32)
    self._last_frame = np.ones((self.contract.output_height, self.contract.output_width), dtype=np.float32)
    self._last_stack = constant_depth_stack(1.0)
    self._last_baseline = None
    self._last_visibility = None
    self._best_visibility = None
    self._visibility_samples = 0

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
    self._renderer = None
    self._render_data = None

  def _ensure_renderer(self) -> None:
    if self._renderer is not None:
      return
    if not os.environ.get("MUJOCO_GL") and not (
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
      os.environ["MUJOCO_GL"] = "egl"
    import mujoco

    sim = self.env.unwrapped.sim
    self._mujoco = mujoco
    self._render_data = mujoco.MjData(sim.mj_model)
    self._renderer = mujoco.Renderer(
      sim.mj_model,
      height=self.contract.raw_height,
      width=self.contract.raw_width,
    )
    self._renderer.enable_depth_rendering()
    self._camera_name = self._resolve_camera_name(sim.mj_model)

  def _resolve_camera_name(self, model: Any) -> str:
    assert self._mujoco is not None
    camera_names = [
      self._mujoco.mj_id2name(model, self._mujoco.mjtObj.mjOBJ_CAMERA, idx)
      for idx in range(model.ncam)
    ]
    for name in camera_names:
      if name == self.contract.camera_name or name.endswith("/" + self.contract.camera_name):
        return name
    raise RuntimeError(
      f"MuJoCo model does not contain camera {self.contract.camera_name!r}; "
      f"available cameras: {camera_names}"
    )

  def _sync_env_state_to_render_data(self) -> None:
    assert self._mujoco is not None
    assert self._render_data is not None
    sim = self.env.unwrapped.sim
    data = self._render_data
    data.qpos[:] = sim.data.qpos[0].cpu().numpy()
    data.qvel[:] = sim.data.qvel[0].cpu().numpy()
    if sim.mj_model.nmocap > 0:
      data.mocap_pos[:] = sim.data.mocap_pos[0].cpu().numpy()
      data.mocap_quat[:] = sim.data.mocap_quat[0].cpu().numpy()
    data.xfrc_applied[:] = sim.data.xfrc_applied[0].cpu().numpy()
    self._mujoco.mj_forward(sim.mj_model, data)

  def _render_raw_depth(self) -> np.ndarray:
    self._ensure_renderer()
    assert self._renderer is not None
    assert self._render_data is not None
    assert self._camera_name is not None
    self._sync_env_state_to_render_data()
    self._renderer.update_scene(self._render_data, camera=self._camera_name)
    return np.asarray(self._renderer.render(), dtype=np.float32)

  def _normalize_and_crop(self, raw_depth_m: np.ndarray) -> np.ndarray:
    depth_min, depth_max = self.contract.depth_range
    output_min, output_max = self.contract.output_range
    normalized = (np.asarray(raw_depth_m, dtype=np.float32) - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = output_min + normalized * (output_max - output_min)
    self._last_raw_frame = normalized.astype(np.float32, copy=True)
    top, bottom, left, right = self.contract.crop_region
    cropped = normalized[
      top : self.contract.raw_height - bottom,
      left : self.contract.raw_width - right,
    ]
    expected = (self.contract.output_height, self.contract.output_width)
    if cropped.shape != expected:
      raise RuntimeError(f"mujoco depth frame shape {cropped.shape}; expected {expected}")
    return cropped.astype(np.float32, copy=False)

  def latest_frame(self, frame_kind: str = "policy") -> np.ndarray:
    if frame_kind == "raw":
      return self._last_raw_frame.copy()
    return self._last_frame.copy()

  def _append_source_frame(self, frame: np.ndarray) -> None:
    if not self._source_history:
      for _ in range(self.contract.history_source_length):
        self._source_history.append(frame.copy())
    else:
      self._source_history.append(frame.copy())

  def _compose_history_stack(self) -> np.ndarray:
    if not self._source_history:
      return self._last_stack.copy()
    latest_index = len(self._source_history) - 1
    frames: list[np.ndarray] = []
    for output_index in reversed(range(self.contract.num_output_frames)):
      source_index = latest_index - output_index * self.contract.history_skip_frames
      source_index = max(0, source_index)
      frames.append(self._source_history[source_index])
    stack = np.stack(frames, axis=0).astype(np.float32, copy=False)
    if stack.shape != self.contract.depth_shape:
      raise RuntimeError(f"mujoco depth stack shape {stack.shape}; expected {self.contract.depth_shape}")
    return stack

  def _visibility_against_flat_baseline(
    self,
    frame: np.ndarray,
    adapter: "ParkourObservationAdapter",
  ) -> dict[str, Any]:
    baseline = self._flat_baseline._render_frame(adapter)
    self._last_baseline = baseline
    delta = frame - baseline
    abs_delta = np.abs(delta)
    roi_specs = {
      "block_roi": (slice(6, 15), slice(8, 24)),
      "gap_roi": (slice(10, 18), slice(10, 22)),
    }
    rois: dict[str, Any] = {}
    for name, (row_slice, col_slice) in roi_specs.items():
      roi_delta = abs_delta[row_slice, col_slice]
      rois[name] = {
        "rows": [row_slice.start, row_slice.stop],
        "cols": [col_slice.start, col_slice.stop],
        "mean_abs_delta": float(np.mean(roi_delta)),
        "max_abs_delta": float(np.max(roi_delta)),
      }
    visibility = {
      "baseline": "analytic-flat-ground",
      "threshold": 0.01,
      "global_mean_abs_delta": float(np.mean(abs_delta)),
      "global_max_abs_delta": float(np.max(abs_delta)),
      "rois": rois,
      "objective_pass": bool(
        rois["block_roi"]["max_abs_delta"] >= 0.01
        and rois["gap_roi"]["max_abs_delta"] >= 0.01
      ),
    }
    return visibility

  def _record_visibility(self, visibility: dict[str, Any]) -> None:
    self._visibility_samples += 1
    visibility = dict(visibility)
    visibility["sample_index"] = self._visibility_samples
    if self._best_visibility is None:
      self._best_visibility = visibility
      return
    current_score = (
      float(visibility.get("global_max_abs_delta", 0.0)),
      float(visibility.get("global_mean_abs_delta", 0.0)),
    )
    best_score = (
      float(self._best_visibility.get("global_max_abs_delta", 0.0)),
      float(self._best_visibility.get("global_mean_abs_delta", 0.0)),
    )
    if bool(visibility.get("objective_pass")) and not bool(self._best_visibility.get("objective_pass")):
      self._best_visibility = visibility
    elif bool(visibility.get("objective_pass")) == bool(self._best_visibility.get("objective_pass")) and current_score > best_score:
      self._best_visibility = visibility

  def _save_previews(self, frame: np.ndarray) -> None:
    if self.debug_dir is None:
      self._preview_paths = {}
      return
    latest = self.debug_dir / "mujoco_depth_latest.pgm"
    _write_grayscale_pgm(latest, frame)
    paths = {"latest": str(latest)}
    if self._last_baseline is not None:
      baseline = self.debug_dir / "mujoco_depth_flat_baseline.pgm"
      delta = self.debug_dir / "mujoco_depth_abs_delta.pgm"
      _write_grayscale_pgm(baseline, self._last_baseline)
      _write_grayscale_pgm(delta, np.abs(frame - self._last_baseline))
      paths.update({"flat_baseline": str(baseline), "abs_delta": str(delta)})
    self._preview_paths = paths

  def stack(self, adapter: "ParkourObservationAdapter | None" = None) -> np.ndarray:
    if adapter is None:
      return self._last_stack.copy()
    raw_depth_m = self._render_raw_depth()
    frame = self._normalize_and_crop(raw_depth_m)
    self._last_frame = frame
    self._last_visibility = self._visibility_against_flat_baseline(frame, adapter)
    self._record_visibility(self._last_visibility)
    self._append_source_frame(frame)
    self._last_stack = self._compose_history_stack()
    self._save_previews(frame)
    return self._last_stack.copy()

  def diagnostics(self) -> dict[str, Any]:
    return {
      "mode": "mujoco",
      "shape": list(self._last_stack.shape),
      "size": int(self._last_stack.size),
      "stats": vector_stats(self._last_stack),
      "camera": {
        "camera_name": self.contract.camera_name,
        "resolved_camera_name": self._camera_name,
        "raw_resolution": list(self.contract.raw_resolution),
        "crop_region": list(self.contract.crop_region),
        "output_resolution": list(self.contract.output_resolution),
        "camera_pose": dict(self.contract.camera_pose),
      },
      "normalization": {
        "depth_range": list(self.contract.depth_range),
        "output_range": list(self.contract.output_range),
        "direction": "near=dark far=bright",
      },
      "history": {
        "history_source_length": self.contract.history_source_length,
        "history_skip_frames": self.contract.history_skip_frames,
        "num_output_frames": self.contract.num_output_frames,
        "seeded": bool(self._source_history),
        "source_frames": len(self._source_history),
      },
      "visibility": self._last_visibility,
      "visibility_best": self._best_visibility,
      "visibility_samples": self._visibility_samples,
      "preview_paths": dict(self._preview_paths),
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

  def set_command(self, command: tuple[float, float, float] | np.ndarray) -> None:
    """Update the velocity command used by proprio history and env command term."""
    self.command = np.asarray(command, dtype=np.float32)

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


def make_depth_provider(
  depth_mode: str,
  constant_depth: float,
  env: Any | None = None,
  debug_dir: Path | None = None,
) -> ConstantDepthProvider | FlatGroundDepthProvider | MujocoRendererDepthProvider:
  if depth_mode == "constant":
    return ConstantDepthProvider(constant_depth)
  if depth_mode == "flat-ground":
    return FlatGroundDepthProvider()
  if depth_mode == "mujoco":
    if env is None:
      raise ValueError("--depth-mode mujoco requires an env-backed renderer")
    return MujocoRendererDepthProvider(env=env, debug_dir=debug_dir)
  raise ValueError(f"Unsupported parkour depth mode: {depth_mode!r}")


def assert_depth_contract(depth_stack: np.ndarray) -> None:
  if depth_stack.size != DEPTH_SIZE:
    raise RuntimeError(f"Depth stack size {depth_stack.size}; expected {DEPTH_SIZE}")
  if not np.isfinite(depth_stack).all():
    raise RuntimeError("Depth stack contains NaN/Inf")
