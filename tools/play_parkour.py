from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

DEFAULT_TASK = "Unitree-G1-Parkour"
DEFAULT_POLICY_DIR = Path("deploy/robots/g1_parkour/config/policy/parkour/v0")
DEFAULT_DEPTH_MODE = "mujoco"
DEFAULT_VIEWER = "native"
DEFAULT_COMMAND_MODE = "terrain-route"
DEFAULT_TERRAIN_ROUTE_SPEED = 0.25
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_VIDEO_FRAME_RATE = 30.0


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Run the exported depth-conditioned Unitree G1 parkour ONNX policy in a "
      "MuJoCo parkour harness. By default this opens the native MuJoCo viewer, "
      "renders the policy depth camera, loads Unitree-G1-Parkour, and follows "
      "the terrain route waypoints."
    )
  )
  parser.add_argument("--task", default=DEFAULT_TASK, help="MJLab task id to load.")
  parser.add_argument(
    "--policy-dir",
    type=Path,
    default=DEFAULT_POLICY_DIR,
    help="Policy bundle root containing exported/ and params/deploy.yaml.",
  )
  parser.add_argument(
    "--exported-dir",
    type=Path,
    help="Override directory containing actor.onnx and 0-depth_encoder.onnx.",
  )
  parser.add_argument(
    "--depth-mode",
    choices=("constant", "flat-ground", "mujoco"),
    default=DEFAULT_DEPTH_MODE,
    help=(
      "Depth source. 'constant' isolates walking, 'flat-ground' analytically "
      "renders the flat floor through the training camera contract, and "
      "'mujoco' renders the real MuJoCo parkour_depth_camera with the deploy "
      "crop/range/history contract."
    ),
  )
  parser.add_argument(
    "--constant-depth",
    type=float,
    default=0.5,
    help=(
      "Normalized constant depth value in [0, 1] for 8x18x32 history. "
      "Default 0.5 is the gray flat-debug ablation; 1.0 represents far/white."
    ),
  )
  parser.add_argument("--command-x", type=float, default=0.25, help="Fixed forward command in m/s.")
  parser.add_argument("--command-y", type=float, default=0.0, help="Fixed lateral command in m/s.")
  parser.add_argument("--command-yaw", type=float, default=0.0, help="Fixed yaw command in rad/s.")
  parser.add_argument(
    "--command-mode",
    choices=("fixed", "terrain-route"),
    default=DEFAULT_COMMAND_MODE,
    help=(
      "Velocity-command source. 'fixed' uses --command-x/y/yaw. "
      "'terrain-route' steers the command along g1_parkour_route_waypoints "
      "from the loaded task so the robot follows the terrain asset sequence."
    ),
  )
  parser.add_argument(
    "--terrain-route-speed",
    type=float,
    default=DEFAULT_TERRAIN_ROUTE_SPEED,
    help=(
      "Walking speed in m/s for --command-mode terrain-route. "
      "--command-x remains the fixed-mode forward command."
    ),
  )
  parser.add_argument(
    "--terrain-route-lookahead",
    type=float,
    default=1.0,
    help="Lookahead distance in meters for --command-mode terrain-route.",
  )
  parser.add_argument(
    "--terrain-route-max-lateral-speed",
    type=float,
    default=0.35,
    help="Body-frame lateral velocity clamp for terrain-route commands.",
  )
  parser.add_argument(
    "--terrain-route-max-yaw-rate",
    type=float,
    default=0.8,
    help="Yaw-rate clamp for terrain-route commands.",
  )
  parser.add_argument(
    "--terrain-route-yaw-gain",
    type=float,
    default=1.5,
    help="Heading-error gain for terrain-route yaw commands.",
  )
  parser.add_argument(
    "--policy-frame",
    choices=("mjlab", "deploy-align"),
    default="mjlab",
    help=(
      "Body-vector frame for base_ang_vel/projected_gravity. 'mjlab' uses "
      "MuJoCo/MJLab values directly (upright gravity ~= [0,0,-1]); "
      "'deploy-align' applies the real-robot -90deg-Y lowstate adapter."
    ),
  )
  parser.add_argument(
    "--joint-order",
    choices=("isaac", "policy", "robot"),
    default="isaac",
    help=(
      "Order used for joint_pos/joint_vel proprio slices. 'isaac' is the "
      "training/ONNX actor order, 'policy' is deploy.yaml motor order, and "
      "'robot' is MuJoCo entity order."
    ),
  )
  parser.add_argument(
    "--action-order",
    choices=("isaac", "policy", "env"),
    default="isaac",
    help=(
      "Interpret actor output as training/ONNX order ('isaac'), deploy.yaml "
      "motor order ('policy'), or env action-manager order ('env')."
    ),
  )
  parser.add_argument("--device", default="cpu", help="Torch/MJLab device, e.g. cpu or cuda:0.")
  parser.add_argument("--num-envs", type=int, default=1, help="Number of envs; only env 0 is diagnosed.")
  parser.add_argument(
    "--agent",
    choices=("policy", "zero"),
    default="policy",
    help="Use the ONNX policy or a zero-action hold baseline for asset/init diagnostics.",
  )
  parser.add_argument(
    "--max-seconds",
    type=float,
    help=(
      "Validation/viewer duration. Default auto-computes enough time for the "
      "terrain route endpoint, or 20s when no route is available."
    ),
  )
  parser.add_argument("--max-steps", type=int, help="Override validation step count.")
  parser.add_argument(
    "--walk-distance",
    type=float,
    help=(
      "Forward displacement acceptance target. Default is the route endpoint "
      "distance, or 5m when no route is available."
    ),
  )
  parser.add_argument(
    "--startup-blend-seconds",
    type=float,
    default=1.0,
    help="Linearly blend policy actions in from zero during startup, matching deploy safety behavior.",
  )
  parser.add_argument(
    "--action-clip",
    type=float,
    help="Optional symmetric raw actor action clip before action scaling, for ablation diagnostics.",
  )
  parser.add_argument(
    "--action-gain",
    type=float,
    default=1.0,
    help="Optional multiplier on raw actor actions before clipping/scaling, for MuJoCo transfer ablations.",
  )
  parser.add_argument(
    "--action-delay-steps",
    type=int,
    default=0,
    help=(
      "Optional N-step delay between policy output and env joint target. "
      "Training used randomized delayed PD actuators; 0 keeps the direct debug path."
    ),
  )
  parser.add_argument("--fall-height", type=float, default=0.45, help="Independent fall height threshold.")
  parser.add_argument(
    "--bad-gravity-z",
    type=float,
    default=-0.25,
    help="Fall if raw projected gravity z becomes greater than this value.",
  )
  parser.add_argument("--check-contract", action="store_true", help="Check deploy YAML and ONNX metadata, then exit.")
  parser.add_argument("--smoke-step", action="store_true", help="Run one synthetic ONNX step with zero proprio and constant depth.")
  parser.add_argument("--validate-walk", action="store_true", help="Run the MuJoCo fixed-command walking validation loop.")
  parser.add_argument(
    "--depth-contract-only",
    action="store_true",
    help=(
      "For short renderer-depth checks, accept finite depth/camera contract "
      "diagnostics without treating the run as traversal success."
    ),
  )
  parser.add_argument(
    "--viewer",
    choices=("none", "native"),
    default=DEFAULT_VIEWER,
    help="Open a realtime viewer instead of the headless validation loop. 'native' requires DISPLAY or WAYLAND_DISPLAY.",
  )
  parser.add_argument(
    "--viewer-frame-rate",
    type=float,
    default=60.0,
    help="Target render frame rate for --viewer native.",
  )
  parser.add_argument(
    "--viewer-run-until-closed",
    action="store_true",
    help="Keep --viewer native running until the MuJoCo window is closed; otherwise --max-seconds/--max-steps bounds the run.",
  )
  parser.add_argument("--debug-parkour", action="store_true", help="Print detailed first-step and periodic diagnostics.")
  parser.add_argument(
    "--video",
    action="store_true",
    help="Record a 1080p MuJoCo video while playing.",
  )
  parser.add_argument(
    "--video-dir",
    type=Path,
    help=(
      "Directory for --video output. Defaults to the exported model directory "
      "containing actor.onnx and 0-depth_encoder.onnx."
    ),
  )
  parser.add_argument(
    "--video-width",
    type=int,
    default=DEFAULT_VIDEO_WIDTH,
    help="Video width in pixels. Default 1920 for 1080p output.",
  )
  parser.add_argument(
    "--video-height",
    type=int,
    default=DEFAULT_VIDEO_HEIGHT,
    help="Video height in pixels. Default 1080.",
  )
  parser.add_argument(
    "--video-frame-rate",
    type=float,
    default=DEFAULT_VIDEO_FRAME_RATE,
    help="Video frame rate in frames per second.",
  )
  parser.add_argument(
    "--gait-record-jsonl",
    type=Path,
    help=(
      "Write per-policy-step gait/action/joint samples as JSONL for Python vs "
      "C++/DDS parity analysis."
    ),
  )
  parser.add_argument(
    "--gait-record-every",
    type=int,
    default=1,
    help="Record one gait sample every N policy steps when --gait-record-jsonl is set.",
  )
  parser.add_argument("--diagnostic-json", type=Path, help="Write diagnostics summary JSON to this path.")
  parser.add_argument(
    "--depth-debug-dir",
    type=Path,
    help=(
      "Optional directory for renderer-depth previews/stat artifacts. "
      "Currently used by --depth-mode mujoco."
    ),
  )
  parser.set_defaults(depth_viewer=True)
  parser.add_argument(
    "--depth-viewer",
    dest="depth_viewer",
    action="store_true",
    help=(
      "Open a live grayscale window for the current normalized depth image. "
      "Use with --depth-mode mujoco to inspect the real parkour_depth_camera stream."
    ),
  )
  parser.add_argument(
    "--no-depth-viewer",
    dest="depth_viewer",
    action="store_false",
    help="Disable the live policy-depth camera window.",
  )
  parser.add_argument(
    "--depth-viewer-frame",
    choices=("policy", "raw"),
    default="policy",
    help=(
      "Depth image shown by --depth-viewer: 'policy' is the cropped 18x32 "
      "policy input, 'raw' is the normalized 64x36 renderer frame when available."
    ),
  )
  parser.add_argument(
    "--depth-viewer-frame-rate",
    type=float,
    default=15.0,
    help="Maximum refresh rate for --depth-viewer.",
  )
  parser.add_argument(
    "--no-terminations",
    action="store_true",
    help="Disable env terminations for debugging; independent fall checks still run.",
  )
  return parser


def bootstrap_tasks() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401


def _json_default(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  return str(value)


def _write_diagnostics(path: Path | None, payload: dict[str, Any]) -> None:
  if path is None:
    return
  resolved = path.expanduser()
  resolved.parent.mkdir(parents=True, exist_ok=True)
  resolved.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _print_json(payload: dict[str, Any]) -> None:
  print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _prepare_mujoco_renderer_env(args: argparse.Namespace) -> None:
  needs_offscreen = args.depth_mode == "mujoco" or bool(getattr(args, "video", False))
  if needs_offscreen and not os.environ.get("MUJOCO_GL") and not (
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
  ):
    # Must be set before imports that transitively load ``mujoco``.  MJLab task
    # and runner packages can import MuJoCo before the renderer provider exists.
    os.environ["MUJOCO_GL"] = "egl"


class LiveDepthViewer:
  """Small optional Matplotlib window for live normalized depth inspection."""

  def __init__(
    self,
    *,
    enabled: bool,
    title: str,
    frame_kind: str,
    frame_rate: float,
  ) -> None:
    self.enabled = enabled
    self.title = title
    self.frame_kind = frame_kind
    self.min_interval = 1.0 / max(frame_rate, 1.0)
    self._last_update = 0.0
    self._plt: Any | None = None
    self._figure: Any | None = None
    self._axes: Any | None = None
    self._image: Any | None = None
    if not enabled:
      return
    _require_graphical_display()
    try:
      import matplotlib.pyplot as plt
    except ImportError as exc:
      raise RuntimeError("--depth-viewer requires matplotlib") from exc

    self._plt = plt
    plt.ion()
    self._figure, self._axes = plt.subplots(num=title)
    self._axes.set_axis_off()
    self._figure.canvas.manager.set_window_title(title)

  def update(self, frame: np.ndarray, diagnostics: dict[str, Any] | None = None) -> None:
    if not self.enabled or self._plt is None or self._figure is None or self._axes is None:
      return
    if not self._plt.fignum_exists(self._figure.number):
      self.enabled = False
      return
    now = time.monotonic()
    if now - self._last_update < self.min_interval:
      return
    self._last_update = now
    image = np.clip(np.asarray(frame, dtype=np.float32), 0.0, 1.0)
    if self._image is None:
      self._image = self._axes.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
      )
    else:
      self._image.set_data(image)
    stats = {
      "min": float(np.nanmin(image)),
      "max": float(np.nanmax(image)),
      "mean": float(np.nanmean(image)),
    }
    camera_name = ""
    if diagnostics:
      camera = diagnostics.get("camera") or {}
      camera_name = camera.get("resolved_camera_name") or camera.get("camera_name") or ""
    self._axes.set_title(
      f"{self.title} [{self.frame_kind}] {camera_name} "
      f"min={stats['min']:.3f} mean={stats['mean']:.3f} max={stats['max']:.3f}\n"
      "0=near/dark, 1=far/bright"
    )
    self._figure.canvas.draw_idle()
    self._plt.pause(0.001)

  def close(self) -> None:
    if self._plt is not None and self._figure is not None and self._plt.fignum_exists(self._figure.number):
      self._plt.close(self._figure)


def _depth_display_frame(depth_provider: Any, fallback_stack: np.ndarray, frame_kind: str) -> np.ndarray:
  latest_frame = getattr(depth_provider, "latest_frame", None)
  if callable(latest_frame):
    return latest_frame(frame_kind)
  return np.asarray(fallback_stack[-1], dtype=np.float32)


def _wrap_pi(angle: float) -> float:
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_wxyz(root_quat: Sequence[float]) -> float:
  w, x, y, z = [float(value) for value in root_quat[:4]]
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ParkourTerrainRouteFollower:
  """Convert terrain route waypoints into policy velocity commands."""

  def __init__(
    self,
    *,
    waypoints: Sequence[Sequence[float]],
    speed: float,
    lookahead: float,
    max_lateral_speed: float,
    max_yaw_rate: float,
    yaw_gain: float,
  ) -> None:
    if len(waypoints) < 2:
      raise ValueError("terrain route requires at least two waypoints")
    self.waypoints = tuple((float(point[0]), float(point[1])) for point in waypoints)
    self.speed = float(speed)
    self.lookahead = max(0.05, float(lookahead))
    self.max_lateral_speed = abs(float(max_lateral_speed))
    self.max_yaw_rate = abs(float(max_yaw_rate))
    self.yaw_gain = float(yaw_gain)

  def _target_waypoint(self, x: float) -> tuple[int, tuple[float, float]]:
    target_x = x + self.lookahead
    for index, waypoint in enumerate(self.waypoints[1:], start=1):
      if waypoint[0] >= target_x:
        return index, waypoint
    return len(self.waypoints) - 1, self.waypoints[-1]

  def command(
    self,
    *,
    base_pos: Sequence[float],
    root_quat: Sequence[float],
  ) -> tuple[tuple[float, float, float], dict[str, Any]]:
    pos_x = float(base_pos[0])
    pos_y = float(base_pos[1])
    yaw = _yaw_from_wxyz(root_quat)
    target_index, target = self._target_waypoint(pos_x)
    route_completed = target_index == len(self.waypoints) - 1 and pos_x >= target[0]
    if route_completed:
      diagnostics = {
        "target_index": target_index,
        "target_waypoint": [target[0], target[1]],
        "base_xy": [pos_x, pos_y],
        "distance_to_target": 0.0,
        "yaw": yaw,
        "desired_heading": yaw,
        "yaw_error": 0.0,
        "command": [0.0, 0.0, 0.0],
        "route_completed": True,
      }
      return (0.0, 0.0, 0.0), diagnostics
    delta_x = target[0] - pos_x
    delta_y = target[1] - pos_y
    distance = max(1.0e-6, math.hypot(delta_x, delta_y))
    desired_heading = math.atan2(delta_y, delta_x)
    yaw_error = _wrap_pi(desired_heading - yaw)

    desired_world_x = self.speed * delta_x / distance
    desired_world_y = self.speed * delta_y / distance
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    body_x = cos_yaw * desired_world_x + sin_yaw * desired_world_y
    body_y = -sin_yaw * desired_world_x + cos_yaw * desired_world_y

    command_x = float(np.clip(body_x, 0.05, max(0.05, self.speed)))
    command_y = float(np.clip(body_y, -self.max_lateral_speed, self.max_lateral_speed))
    command_yaw = float(np.clip(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate))
    diagnostics = {
      "target_index": target_index,
      "target_waypoint": [target[0], target[1]],
      "base_xy": [pos_x, pos_y],
      "distance_to_target": distance,
      "yaw": yaw,
      "desired_heading": desired_heading,
      "yaw_error": yaw_error,
      "command": [command_x, command_y, command_yaw],
      "route_completed": False,
    }
    return (command_x, command_y, command_yaw), diagnostics


def _route_waypoints_from_env(env: Any) -> tuple[tuple[float, float], ...]:
  cfg = getattr(env, "cfg", None)
  waypoints = getattr(cfg, "g1_parkour_route_waypoints", ()) if cfg is not None else ()
  return tuple((float(point[0]), float(point[1])) for point in waypoints)


def _route_endpoint_distance(env: Any) -> float | None:
  waypoints = _route_waypoints_from_env(env)
  if len(waypoints) < 2:
    return None
  return max(0.0, waypoints[-1][0] - waypoints[0][0])


def _resolve_walk_distance(args: argparse.Namespace, env: Any) -> float:
  if args.walk_distance is not None:
    return float(args.walk_distance)
  route_distance = _route_endpoint_distance(env)
  return route_distance if route_distance is not None else 5.0


def _resolve_max_seconds(args: argparse.Namespace, env: Any) -> float:
  if args.max_seconds is not None:
    return float(args.max_seconds)
  route_distance = _route_endpoint_distance(env)
  if route_distance is None:
    return 20.0
  speed_value = args.terrain_route_speed if args.command_mode == "terrain-route" else args.command_x
  speed = max(abs(float(speed_value)), 0.05)
  return max(20.0, route_distance / speed + 10.0)


def _make_route_follower(
  env: Any,
  args: argparse.Namespace,
) -> ParkourTerrainRouteFollower | None:
  if args.command_mode != "terrain-route":
    return None
  waypoints = _route_waypoints_from_env(env)
  if not waypoints:
    print("[parkour] No g1_parkour_route_waypoints on task; falling back to fixed command.")
    return None
  return ParkourTerrainRouteFollower(
    waypoints=waypoints,
    speed=args.terrain_route_speed,
    lookahead=args.terrain_route_lookahead,
    max_lateral_speed=args.terrain_route_max_lateral_speed,
    max_yaw_rate=args.terrain_route_max_yaw_rate,
    yaw_gain=args.terrain_route_yaw_gain,
  )


def _apply_play_command(
  adapter: Any,
  *,
  route_follower: ParkourTerrainRouteFollower | None,
  fixed_command: tuple[float, float, float],
) -> dict[str, Any] | None:
  if route_follower is None:
    adapter.set_command(fixed_command)
    adapter.set_fixed_command()
    return None
  fall_signals = adapter.fall_signals()
  command, diagnostics = route_follower.command(
    base_pos=fall_signals["base_pos"],
    root_quat=fall_signals["root_quat"],
  )
  adapter.set_command(command)
  adapter.set_fixed_command()
  return diagnostics


def _resolve_video_output_path(args: argparse.Namespace, paths: Any) -> Path:
  output_dir = args.video_dir if args.video_dir is not None else Path(paths.exported_dir)
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  return output_dir.expanduser() / f"play_parkour_{timestamp}.mp4"


class GaitJsonlRecorder:
  """JSONL recorder for comparing Python play and C++/DDS gait pipelines."""

  def __init__(
    self,
    *,
    path: Path | None,
    every: int,
    source: str,
  ) -> None:
    self.path = path.expanduser() if path is not None else None
    self.every = max(1, int(every))
    self.source = source
    self.samples = 0
    self._file: Any | None = None

  @property
  def enabled(self) -> bool:
    return self.path is not None

  def _ensure_open(self) -> None:
    if not self.enabled or self._file is not None:
      return
    assert self.path is not None
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._file = self.path.open("w", encoding="utf-8")

  @staticmethod
  def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim > 1:
      arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)

  @staticmethod
  def _tolist(value: Any) -> list[float]:
    return [float(item) for item in GaitJsonlRecorder._array(value).tolist()]

  @staticmethod
  def _vector_by_joint_name(
    *,
    names: Sequence[str],
    values: np.ndarray,
    target_names: Sequence[str],
  ) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = [name for name in target_names if name not in name_to_index]
    if missing:
      raise RuntimeError(f"Cannot record gait vector; missing joints: {missing}")
    arr = GaitJsonlRecorder._array(values)
    return np.asarray([arr[name_to_index[name]] for name in target_names], dtype=np.float32)

  def record_python_step(
    self,
    *,
    step: int,
    elapsed: float,
    adapter: Any,
    contract: Any,
    depth: np.ndarray,
    raw_policy_action: np.ndarray,
    applied_policy_action: np.ndarray,
    env_action: np.ndarray,
    latest_policy_diag: Mapping[str, Any],
    latest_frame_diag: Any,
    latest_route_diag: Mapping[str, Any] | None,
    fall_signals: Mapping[str, Any],
  ) -> None:
    if not self.enabled or step % self.every != 0:
      return
    self._ensure_open()
    assert self._file is not None

    from src.parkour.contract import ONNX_POLICY_JOINT_NAMES, TRAINING_JOINT_NAMES, vector_stats

    robot_joint_names = tuple(adapter.robot.joint_names)
    training_joint_names = tuple(TRAINING_JOINT_NAMES)
    policy_joint_names = tuple(ONNX_POLICY_JOINT_NAMES)
    training_idx = [robot_joint_names.index(name) for name in training_joint_names]
    policy_idx = [robot_joint_names.index(name) for name in policy_joint_names]
    joint_pos = adapter.robot.data.joint_pos[:, torch.tensor(training_idx, device=adapter.env.device)]
    joint_vel = adapter.robot.data.joint_vel[:, torch.tensor(training_idx, device=adapter.env.device)]
    policy_joint_pos = adapter.robot.data.joint_pos[:, torch.tensor(policy_idx, device=adapter.env.device)]
    policy_joint_vel = adapter.robot.data.joint_vel[:, torch.tensor(policy_idx, device=adapter.env.device)]

    if adapter.action_order == "isaac":
      action_joint_names = policy_joint_names
    elif adapter.action_order == "policy":
      action_joint_names = training_joint_names
    else:
      action_joint_names = tuple(adapter.env_action_target_names)
    raw_action_deploy = self._vector_by_joint_name(
      names=action_joint_names,
      values=raw_policy_action,
      target_names=training_joint_names,
    )
    applied_action_deploy = self._vector_by_joint_name(
      names=action_joint_names,
      values=applied_policy_action,
      target_names=training_joint_names,
    )

    env_target_names = tuple(adapter.env_action_target_names)
    env_action_flat = self._array(env_action)
    env_action_deploy = self._vector_by_joint_name(
      names=env_target_names,
      values=env_action_flat,
      target_names=training_joint_names,
    )
    scale_by_joint = contract.action_scale_by_joint
    offset_by_joint = contract.action_offset_by_joint
    target_q_deploy = np.asarray(
      [
        env_action_deploy[index] * scale_by_joint[name] + offset_by_joint[name]
        for index, name in enumerate(training_joint_names)
      ],
      dtype=np.float32,
    )

    payload = {
      "source": self.source,
      "step": int(step),
      "elapsed_seconds": float(elapsed),
      "joint_order": {
        "deploy": list(training_joint_names),
        "policy": list(policy_joint_names),
        "env_action": list(env_target_names),
      },
      "command": self._tolist(getattr(adapter, "command", np.zeros(3, dtype=np.float32))),
      "base_ang_vel": (
        list(latest_frame_diag.policy_base_ang_vel)
        if latest_frame_diag is not None
        else []
      ),
      "projected_gravity": (
        list(latest_frame_diag.policy_projected_gravity)
        if latest_frame_diag is not None
        else []
      ),
      "base_pos": list(fall_signals.get("base_pos", [])),
      "root_quat": list(fall_signals.get("root_quat", [])),
      "joint_pos_deploy_order": self._tolist(joint_pos),
      "joint_vel_deploy_order": self._tolist(joint_vel),
      "joint_pos_policy_order": self._tolist(policy_joint_pos),
      "joint_vel_policy_order": self._tolist(policy_joint_vel),
      "raw_action_policy_order": self._tolist(raw_policy_action),
      "applied_action_policy_order": self._tolist(applied_policy_action),
      "raw_action_deploy_order": self._tolist(raw_action_deploy),
      "applied_action_deploy_order": self._tolist(applied_action_deploy),
      "env_action_deploy_order": self._tolist(env_action_deploy),
      "target_q_deploy_order": self._tolist(target_q_deploy),
      "depth_stats": vector_stats(depth),
      "policy": {
        "action_stats": latest_policy_diag.get("action_stats"),
        "applied_action_stats": latest_policy_diag.get("applied_action_stats"),
        "proprio_stats": latest_policy_diag.get("proprio_stats"),
        "actor_input_stats": latest_policy_diag.get("actor_input_stats"),
        "latent_stats": latest_policy_diag.get("latent_stats"),
      },
      "route": dict(latest_route_diag) if latest_route_diag is not None else None,
    }
    self._file.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")
    self.samples += 1

  def diagnostics(self) -> dict[str, Any] | None:
    if not self.enabled:
      return None
    return {
      "path": str(self.path) if self.path is not None else None,
      "every": self.every,
      "source": self.source,
      "samples": self.samples,
    }

  def close(self) -> None:
    if self._file is not None:
      self._file.close()
      self._file = None


class ParkourVideoRecorder:
  """Streaming MuJoCo video recorder for play_parkour."""

  def __init__(
    self,
    *,
    enabled: bool,
    env: Any,
    output_path: Path | None,
    width: int,
    height: int,
    frame_rate: float,
  ) -> None:
    self.enabled = enabled
    self.env = env
    self.output_path = output_path
    self.width = int(width)
    self.height = int(height)
    self.frame_rate = float(frame_rate)
    self.frame_interval = 1.0 / max(self.frame_rate, 1.0)
    self._last_capture_time = -1.0e9
    self._mujoco: Any | None = None
    self._renderer: Any | None = None
    self._render_data: Any | None = None
    self._camera: Any | None = None
    self._writer: Any | None = None
    self.frame_count = 0

  def _ensure_open(self) -> None:
    if not self.enabled or self._writer is not None:
      return
    if self.output_path is None:
      raise RuntimeError("video output path is required when --video is enabled")
    if self.width <= 0 or self.height <= 0:
      raise ValueError("--video-width and --video-height must be positive")
    if self.frame_rate <= 0.0:
      raise ValueError("--video-frame-rate must be positive")
    if not os.environ.get("MUJOCO_GL") and not (
      os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
      os.environ["MUJOCO_GL"] = "egl"
    import imageio.v2 as imageio
    import mujoco

    sim = self.env.unwrapped.sim
    self._mujoco = mujoco
    self._render_data = mujoco.MjData(sim.mj_model)
    self._renderer = mujoco.Renderer(
      sim.mj_model,
      height=self.height,
      width=self.width,
    )
    self._camera = self._make_tracking_camera(sim.mj_model)
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    self._writer = imageio.get_writer(
      str(self.output_path),
      fps=self.frame_rate,
      codec="libx264",
      quality=8,
      macro_block_size=1,
    )

  def _make_tracking_camera(self, model: Any) -> Any:
    assert self._mujoco is not None
    camera = self._mujoco.MjvCamera()
    body_id = -1
    for candidate in ("torso_link", "robot/torso_link"):
      body_id = self._mujoco.mj_name2id(
        model,
        self._mujoco.mjtObj.mjOBJ_BODY,
        candidate,
      )
      if body_id >= 0:
        break
    if body_id >= 0:
      camera.type = self._mujoco.mjtCamera.mjCAMERA_TRACKING
      camera.trackbodyid = body_id
    else:
      camera.type = self._mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.lookat[:] = (0.0, 0.0, 0.8)
    return camera

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

  def capture(self, sim_time: float) -> None:
    if not self.enabled:
      return
    if sim_time - self._last_capture_time < self.frame_interval:
      return
    self._ensure_open()
    assert self._renderer is not None
    assert self._writer is not None
    self._sync_env_state_to_render_data()
    self._renderer.update_scene(self._render_data, camera=self._camera)
    self._writer.append_data(self._renderer.render())
    self._last_capture_time = sim_time
    self.frame_count += 1

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()
      self._writer = None
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
    self._render_data = None
    self._camera = None

  def diagnostics(self) -> dict[str, Any] | None:
    if not self.enabled:
      return None
    return {
      "path": str(self.output_path) if self.output_path is not None else None,
      "width": self.width,
      "height": self.height,
      "frame_rate": self.frame_rate,
      "frames": self.frame_count,
    }


def _run_native_viewer_loop(
  viewer: Any,
  *,
  num_steps: int | None,
  stop_condition: Any,
  video_recorder: ParkourVideoRecorder,
  sim_time_fn: Any,
) -> None:
  viewer.setup()
  now = time.perf_counter()
  viewer._stats_last_time = now
  viewer._last_tick_time = now
  try:
    while viewer.is_running() and (num_steps is None or viewer._step_count < num_steps):
      if stop_condition():
        break
      if viewer.tick():
        video_recorder.capture(sim_time_fn())
      else:
        time.sleep(0.001)
      viewer._update_stats()
      if stop_condition():
        break
  finally:
    viewer.close()


def _load_contract_and_policy(args: argparse.Namespace):
  from src.parkour.contract import (
    load_deploy_contract,
    resolve_model_paths,
    validate_model_files,
  )
  from src.parkour.onnx_policy import ParkourOnnxPolicy

  paths = resolve_model_paths(policy_dir=args.policy_dir, exported_dir=args.exported_dir)
  validate_model_files(paths)
  contract = load_deploy_contract(paths.deploy_yaml)
  policy = ParkourOnnxPolicy(policy_dir=paths.policy_dir, exported_dir=paths.exported_dir)
  return contract, policy, paths


def run_check_contract(args: argparse.Namespace) -> dict[str, Any]:
  contract, policy, paths = _load_contract_and_policy(args)
  payload = {
    "mode": "check-contract",
    "policy_dir": paths.policy_dir,
    "exported_dir": paths.exported_dir,
    "deploy_yaml": paths.deploy_yaml,
    "onnx": policy.metadata.as_dict(),
    "proprio_size": contract.proprio_size,
    "depth_size": contract.depth_size,
    "actor_input_size": contract.actor_input_size,
    "action_size": contract.action_size,
    "joint_names_head": list(contract.joint_names[:6]),
    "joint_names_tail": list(contract.joint_names[-6:]),
    "status": "ok",
  }
  _print_json(payload)
  return payload


def run_smoke_step(args: argparse.Namespace) -> dict[str, Any]:
  from src.parkour.contract import PROPRIO_SIZE, constant_depth_stack

  contract, policy, paths = _load_contract_and_policy(args)
  proprio = np.zeros(PROPRIO_SIZE, dtype=np.float32)
  depth = constant_depth_stack(args.constant_depth)
  output = policy.act(proprio, depth)
  payload = {
    "mode": "smoke-step",
    "policy_dir": paths.policy_dir,
    "depth_mode": "constant",
    "constant_depth": args.constant_depth,
    "onnx": policy.metadata.as_dict(),
    "depth_shape": list(depth.shape),
    "proprio_size": int(proprio.size),
    "action_shape": list(output.action.shape),
    "action_head": output.action[:10].tolist(),
    "diagnostics": output.diagnostics,
    "status": "ok",
  }
  _print_json(payload)
  return payload


def _load_env(args: argparse.Namespace):
  _prepare_mujoco_renderer_env(args)

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from src.parkour.contract import assert_no_stale_sensor_references

  bootstrap_tasks()
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  if args.no_terminations:
    env_cfg.terminations = {}
    print("[parkour] Env terminations disabled; independent fall checks remain active.")
  assert_no_stale_sensor_references(env_cfg)
  return ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)


def _require_graphical_display() -> None:
  if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    return
  raise RuntimeError(
    "Native viewer requires a graphical display; DISPLAY or WAYLAND_DISPLAY must be set."
  )


class ParkourNativeViewerPolicy:
  """Realtime native-viewer policy adapter for the exported parkour ONNX actor."""

  def __init__(
    self,
    *,
    env: Any,
    contract: Any,
    policy: Any | None,
    depth_provider: Any,
    depth_viewer: LiveDepthViewer | None,
    gait_recorder: GaitJsonlRecorder,
    args: argparse.Namespace,
  ) -> None:
    self.env = env
    self.contract = contract
    self.policy = policy
    self.depth_provider = depth_provider
    self.depth_viewer = depth_viewer
    self.gait_recorder = gait_recorder
    self.args = args
    self.adapter: Any | None = None
    self.step_count = 0
    self.start_x = 0.0
    self.distance_x = 0.0
    self.last_action = np.zeros(contract.action_size, dtype=np.float32)
    self.latest_frame_diag: Any = None
    self.latest_policy_diag: dict[str, Any] = {}
    self.latest_fall_signals: dict[str, Any] = {}
    self.mapping_proof: dict[str, Any] | None = None
    self.action_delay_buffer: deque[np.ndarray] = deque(maxlen=max(1, args.action_delay_steps + 1))
    self.fixed_command = (args.command_x, args.command_y, args.command_yaw)
    self.route_follower = _make_route_follower(env, args)
    self.latest_route_diag: dict[str, Any] | None = None
    self.route_completed = False
    self._has_stepped = False
    self.reset()

  def reset(self) -> None:
    if self.args.action_delay_steps < 0:
      raise ValueError("--action-delay-steps must be non-negative")
    from src.tasks.velocity.rl.parkour_play import ParkourObservationAdapter

    self.adapter = ParkourObservationAdapter(
      self.env,
      self.contract,
      command=(self.args.command_x, self.args.command_y, self.args.command_yaw),
      frame_mode=self.args.policy_frame,
      joint_order=self.args.joint_order,
      action_order=self.args.action_order,
    )
    self.last_action = np.zeros(self.contract.action_size, dtype=np.float32)
    self.latest_route_diag = _apply_play_command(
      self.adapter,
      route_follower=self.route_follower,
      fixed_command=self.fixed_command,
    )
    self.route_completed = bool(
      self.latest_route_diag and self.latest_route_diag.get("route_completed")
    )
    self.latest_frame_diag = self.adapter.warm_start(last_policy_action=self.last_action)
    self.latest_fall_signals = self.adapter.fall_signals()
    self.start_x = float(self.latest_fall_signals["base_pos"][0])
    self.distance_x = 0.0
    self.step_count = 0
    self.mapping_proof = self.adapter.mapping_proof().as_dict()
    self.latest_policy_diag = {}
    self.action_delay_buffer.clear()
    if hasattr(self.depth_provider, "reset"):
      self.depth_provider.reset()
    for _ in range(self.args.action_delay_steps):
      self.action_delay_buffer.append(np.zeros(self.contract.action_size, dtype=np.float32))
    self._has_stepped = False

  def __call__(self, obs: Any) -> torch.Tensor:
    del obs
    from src.tasks.velocity.rl.parkour_play import assert_depth_contract, vector_stats

    if self.adapter is None:
      self.reset()
    assert self.adapter is not None

    # After the previous env.step(), append the current MuJoCo state with the
    # action that was just applied, matching the headless validation history.
    if self._has_stepped:
      self.latest_route_diag = _apply_play_command(
        self.adapter,
        route_follower=self.route_follower,
        fixed_command=self.fixed_command,
      )
      self.route_completed = bool(
        self.latest_route_diag and self.latest_route_diag.get("route_completed")
      )
      self.latest_frame_diag = self.adapter.append_current(last_policy_action=self.last_action)
      self.latest_fall_signals = self.adapter.fall_signals()
      self.distance_x = float(self.latest_fall_signals["base_pos"][0]) - self.start_x

    self.latest_route_diag = _apply_play_command(
      self.adapter,
      route_follower=self.route_follower,
      fixed_command=self.fixed_command,
    )
    self.route_completed = bool(
      self.latest_route_diag and self.latest_route_diag.get("route_completed")
    )
    proprio = self.adapter.proprio()
    depth = self.depth_provider.stack(self.adapter)
    assert_depth_contract(depth)
    if self.depth_viewer is not None:
      self.depth_viewer.update(
        _depth_display_frame(self.depth_provider, depth, self.args.depth_viewer_frame),
        self.depth_provider.diagnostics(),
      )

    if self.route_completed:
      raw_policy_action = np.zeros(self.contract.action_size, dtype=np.float32)
      self.latest_policy_diag = {
        "action_stats": vector_stats(raw_policy_action),
        "actor_input_stats": None,
        "depth_stats": vector_stats(depth),
        "proprio_stats": vector_stats(proprio),
        "route_completed": True,
      }
    elif self.policy is None:
      raw_policy_action = np.zeros(self.contract.action_size, dtype=np.float32)
      self.latest_policy_diag = {
        "action_stats": vector_stats(raw_policy_action),
        "actor_input_stats": None,
        "depth_stats": vector_stats(depth),
        "proprio_stats": vector_stats(proprio),
        "zero_action_baseline": True,
      }
    else:
      output = self.policy.act(proprio, depth)
      raw_policy_action = output.action
      self.latest_policy_diag = dict(output.diagnostics)

    raw_policy_action = raw_policy_action * np.float32(self.args.action_gain)
    blend_alpha = 1.0
    if self.args.startup_blend_seconds > 0.0:
      blend_alpha = min(
        1.0,
        (self.step_count + 1) * self.env.step_dt / self.args.startup_blend_seconds,
      )
    if self.args.action_clip is not None:
      raw_policy_action = np.clip(raw_policy_action, -self.args.action_clip, self.args.action_clip)
    applied_policy_action = raw_policy_action * np.float32(blend_alpha)
    if self.args.action_delay_steps > 0:
      self.action_delay_buffer.append(applied_policy_action.copy())
      env_policy_action = self.action_delay_buffer[0].copy()
    else:
      env_policy_action = applied_policy_action

    self.latest_policy_diag.update(
      {
        "startup_blend_alpha": float(blend_alpha),
        "applied_action_stats": vector_stats(applied_policy_action),
        "env_action_delay_steps": int(self.args.action_delay_steps),
        "env_policy_action_stats": vector_stats(env_policy_action),
      }
    )
    env_action = self.adapter.env_action_from_policy_action(env_policy_action)
    self.gait_recorder.record_python_step(
      step=self.step_count,
      elapsed=self.step_count * self.env.step_dt,
      adapter=self.adapter,
      contract=self.contract,
      depth=depth,
      raw_policy_action=raw_policy_action,
      applied_policy_action=applied_policy_action,
      env_action=env_action,
      latest_policy_diag=self.latest_policy_diag,
      latest_frame_diag=self.latest_frame_diag,
      latest_route_diag=self.latest_route_diag,
      fall_signals=self.latest_fall_signals,
    )
    self.last_action = applied_policy_action.copy()
    self.step_count += 1
    self._has_stepped = True
    return torch.tensor(env_action, dtype=torch.float32, device=self.env.device)

  def diagnostics(self) -> dict[str, Any]:
    return {
      "mode": "viewer",
      "status": "closed",
      "task": self.args.task,
      "agent": self.args.agent,
      "depth": self.depth_provider.diagnostics(),
      "command": [self.args.command_x, self.args.command_y, self.args.command_yaw],
      "command_mode": self.args.command_mode,
      "terrain_route_speed": self.args.terrain_route_speed,
      "route_waypoints": (
        [list(point) for point in self.route_follower.waypoints]
        if self.route_follower is not None
        else None
      ),
      "latest_route": self.latest_route_diag,
      "policy_frame": self.args.policy_frame,
      "joint_order": self.args.joint_order,
      "action_order": self.args.action_order,
      "step": self.step_count,
      "elapsed_seconds": self.step_count * self.env.step_dt,
      "distance_x": self.distance_x,
      "fall_signals": self.latest_fall_signals,
      "latest_frame": self.latest_frame_diag.__dict__ if self.latest_frame_diag is not None else None,
      "latest_policy": self.latest_policy_diag,
      "mapping_proof": self.mapping_proof,
      "gait_record": self.gait_recorder.diagnostics(),
    }


def run_native_viewer(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
  if args.num_envs != 1:
    raise ValueError("--viewer native currently supports --num-envs 1")
  _require_graphical_display()
  _prepare_mujoco_renderer_env(args)

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import NativeMujocoViewer
  from src.parkour.contract import load_deploy_contract, resolve_model_paths
  from src.parkour.onnx_policy import ParkourOnnxPolicy
  from src.tasks.velocity.rl.parkour_play import make_depth_provider

  paths = resolve_model_paths(policy_dir=args.policy_dir, exported_dir=args.exported_dir)
  contract = load_deploy_contract(paths.deploy_yaml)
  policy = None
  if args.agent == "policy":
    policy = ParkourOnnxPolicy(policy_dir=paths.policy_dir, exported_dir=paths.exported_dir)
  raw_env = _load_env(args)
  depth_provider = make_depth_provider(
    args.depth_mode,
    args.constant_depth,
    env=raw_env,
    debug_dir=args.depth_debug_dir,
  )
  depth_viewer = LiveDepthViewer(
    enabled=args.depth_viewer,
    title=f"Parkour depth: {args.task}",
    frame_kind=args.depth_viewer_frame,
    frame_rate=args.depth_viewer_frame_rate,
  )
  video_output_path = _resolve_video_output_path(args, paths) if args.video else None
  video_recorder = ParkourVideoRecorder(
    enabled=args.video,
    env=raw_env,
    output_path=video_output_path,
    width=args.video_width,
    height=args.video_height,
    frame_rate=args.video_frame_rate,
  )
  gait_recorder = GaitJsonlRecorder(
    path=args.gait_record_jsonl,
    every=args.gait_record_every,
    source="python_play",
  )
  viewer_env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
  viewer_policy = ParkourNativeViewerPolicy(
    env=raw_env,
    contract=contract,
    policy=policy,
    depth_provider=depth_provider,
    depth_viewer=depth_viewer,
    gait_recorder=gait_recorder,
    args=args,
  )
  raw_env.reset()
  viewer_policy.reset()
  resolved_max_seconds = _resolve_max_seconds(args, raw_env)
  num_steps = None
  if not args.viewer_run_until_closed:
    num_steps = args.max_steps or max(1, int(round(resolved_max_seconds / raw_env.step_dt)))
  print(
    "[parkour] Launching native MuJoCo viewer. "
    "Close the window to stop; default duration follows the terrain route endpoint."
  )
  try:
    viewer = NativeMujocoViewer(
      viewer_env,
      viewer_policy,
      frame_rate=args.viewer_frame_rate,
      enable_perturbations=True,
    )
    _run_native_viewer_loop(
      viewer,
      num_steps=num_steps,
      stop_condition=lambda: viewer_policy.route_completed and not args.viewer_run_until_closed,
      video_recorder=video_recorder,
      sim_time_fn=lambda: viewer_policy.step_count * raw_env.step_dt,
    )
    payload = viewer_policy.diagnostics()
    payload.update(
      {
        "policy_dir": paths.policy_dir,
        "exported_dir": paths.exported_dir,
        "onnx": policy.metadata.as_dict() if policy is not None else None,
        "viewer": args.viewer,
        "viewer_frame_rate": args.viewer_frame_rate,
        "viewer_run_until_closed": args.viewer_run_until_closed,
        "max_seconds": resolved_max_seconds,
        "max_steps": num_steps,
        "video": video_recorder.diagnostics(),
        "gait_record": gait_recorder.diagnostics(),
      }
    )
    return True, payload
  finally:
    gait_recorder.close()
    video_recorder.close()
    depth_viewer.close()
    if hasattr(depth_provider, "close"):
      depth_provider.close()
    raw_env.close()


def run_validate_walk(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
  _prepare_mujoco_renderer_env(args)
  if args.depth_viewer:
    _require_graphical_display()

  from src.parkour.contract import load_deploy_contract, resolve_model_paths
  from src.parkour.onnx_policy import ParkourOnnxPolicy
  from src.tasks.velocity.rl.parkour_play import (
    ParkourObservationAdapter,
    assert_depth_contract,
    classify_failure,
    make_depth_provider,
    vector_stats,
  )

  paths = resolve_model_paths(policy_dir=args.policy_dir, exported_dir=args.exported_dir)
  contract = load_deploy_contract(paths.deploy_yaml)
  policy = None
  if args.agent == "policy":
    policy = ParkourOnnxPolicy(policy_dir=paths.policy_dir, exported_dir=paths.exported_dir)
  env = _load_env(args)
  depth_provider = make_depth_provider(
    args.depth_mode,
    args.constant_depth,
    env=env,
    debug_dir=args.depth_debug_dir,
  )
  depth_viewer = LiveDepthViewer(
    enabled=args.depth_viewer,
    title=f"Parkour depth: {args.task}",
    frame_kind=args.depth_viewer_frame,
    frame_rate=args.depth_viewer_frame_rate,
  )
  video_output_path = _resolve_video_output_path(args, paths) if args.video else None
  video_recorder = ParkourVideoRecorder(
    enabled=args.video,
    env=env,
    output_path=video_output_path,
    width=args.video_width,
    height=args.video_height,
    frame_rate=args.video_frame_rate,
  )
  gait_recorder = GaitJsonlRecorder(
    path=args.gait_record_jsonl,
    every=args.gait_record_every,
    source="python_play",
  )
  walk_distance = _resolve_walk_distance(args, env)
  max_seconds = _resolve_max_seconds(args, env)
  args.walk_distance = walk_distance
  args.max_seconds = max_seconds

  summary: dict[str, Any] = {
    "mode": "validate-walk",
    "task": args.task,
    "agent": args.agent,
    "policy_dir": paths.policy_dir,
    "exported_dir": paths.exported_dir,
    "depth": depth_provider.diagnostics(),
    "onnx": policy.metadata.as_dict() if policy is not None else None,
    "command": [args.command_x, args.command_y, args.command_yaw],
    "command_mode": args.command_mode,
    "terrain_route_speed": args.terrain_route_speed,
    "policy_frame": args.policy_frame,
    "joint_order": args.joint_order,
    "action_order": args.action_order,
    "max_seconds": max_seconds,
    "walk_distance": walk_distance,
    "depth_contract_only": args.depth_contract_only,
    "startup_blend_seconds": args.startup_blend_seconds,
    "action_clip": args.action_clip,
    "action_gain": args.action_gain,
    "action_delay_steps": args.action_delay_steps,
    "video": video_recorder.diagnostics(),
    "gait_record": gait_recorder.diagnostics(),
  }
  reset_count = 0
  latest_frame_diag: Any = None
  latest_policy_diag: Mapping[str, Any] = {}
  latest_route_diag: dict[str, Any] | None = None
  last_action = np.zeros(contract.action_size, dtype=np.float32)

  try:
    env.reset()
    if hasattr(depth_provider, "reset"):
      depth_provider.reset()
    adapter = ParkourObservationAdapter(
      env,
      contract,
      command=(args.command_x, args.command_y, args.command_yaw),
      frame_mode=args.policy_frame,
      joint_order=args.joint_order,
      action_order=args.action_order,
    )
    fixed_command = (args.command_x, args.command_y, args.command_yaw)
    route_follower = _make_route_follower(env, args)
    summary["route_waypoints"] = (
      [list(point) for point in route_follower.waypoints]
      if route_follower is not None
      else None
    )
    latest_route_diag = _apply_play_command(
      adapter,
      route_follower=route_follower,
      fixed_command=fixed_command,
    )
    latest_frame_diag = adapter.warm_start(last_policy_action=last_action)
    mapping_proof = adapter.mapping_proof().as_dict()
    summary["mapping_proof"] = mapping_proof
    summary["warm_start"] = {
      "history_length": 8,
      "proprio_size": contract.proprio_size,
      "depth_size": contract.depth_size,
      "first_frame": latest_frame_diag.__dict__,
    }

    initial_fall = adapter.fall_signals()
    start_x = float(initial_fall["base_pos"][0])
    if args.debug_parkour:
      print("[parkour] mapping_proof=" + json.dumps(mapping_proof, default=_json_default))
      print("[parkour] warm_start=" + json.dumps(summary["warm_start"], default=_json_default))

    if initial_fall["base_height"] < args.fall_height:
      summary["status"] = "failed"
      summary["failure_reason"] = "initial_fall"
      summary["root_cause"] = classify_failure(
        reason="initial_fall",
        fall_signals=initial_fall,
        depth_mode=args.depth_mode,
      )
      summary["fall_signals"] = initial_fall
      return False, summary

    max_steps = args.max_steps or max(1, int(round(max_seconds / env.step_dt)))
    if args.action_delay_steps < 0:
      raise ValueError("--action-delay-steps must be non-negative")
    action_delay_buffer = deque(
      [np.zeros(contract.action_size, dtype=np.float32) for _ in range(args.action_delay_steps)],
      maxlen=args.action_delay_steps + 1,
    )
    final_fall = initial_fall
    distance = 0.0
    elapsed = 0.0
    for step in range(max_steps):
      latest_route_diag = _apply_play_command(
        adapter,
        route_follower=route_follower,
        fixed_command=fixed_command,
      )
      if latest_route_diag and latest_route_diag.get("route_completed"):
        break
      proprio = adapter.proprio()
      depth = depth_provider.stack(adapter)
      assert_depth_contract(depth)
      depth_viewer.update(
        _depth_display_frame(depth_provider, depth, args.depth_viewer_frame),
        depth_provider.diagnostics(),
      )
      proprio_finite = bool(np.isfinite(proprio).all())
      if not proprio_finite:
        summary["status"] = "failed"
        summary["failure_reason"] = "nonfinite_proprio"
        summary["root_cause"] = classify_failure(
          reason="nonfinite_proprio",
          proprio_finite=False,
          depth_mode=args.depth_mode,
        )
        return False, summary

      if policy is None:
        raw_policy_action = np.zeros(contract.action_size, dtype=np.float32)
        latest_policy_diag = {
          "action_stats": vector_stats(raw_policy_action),
          "actor_input_stats": None,
          "depth_stats": vector_stats(depth),
          "proprio_stats": vector_stats(proprio),
          "zero_action_baseline": True,
        }
      else:
        output = policy.act(proprio, depth)
        latest_policy_diag = output.diagnostics
        raw_policy_action = output.action
      raw_policy_action = raw_policy_action * np.float32(args.action_gain)
      blend_alpha = 1.0
      if args.startup_blend_seconds > 0.0:
        blend_alpha = min(1.0, (step + 1) * env.step_dt / args.startup_blend_seconds)
      if args.action_clip is not None:
        raw_policy_action = np.clip(raw_policy_action, -args.action_clip, args.action_clip)
      applied_policy_action = raw_policy_action * np.float32(blend_alpha)
      if args.action_delay_steps > 0:
        action_delay_buffer.append(applied_policy_action.copy())
        env_policy_action = action_delay_buffer[0].copy()
      else:
        env_policy_action = applied_policy_action
      latest_policy_diag = {
        **dict(latest_policy_diag),
        "startup_blend_alpha": float(blend_alpha),
        "applied_action_stats": vector_stats(applied_policy_action),
        "env_action_delay_steps": int(args.action_delay_steps),
        "env_policy_action_stats": vector_stats(env_policy_action),
      }
      action_finite = bool(np.isfinite(applied_policy_action).all())
      if not action_finite:
        summary["status"] = "failed"
        summary["failure_reason"] = "nonfinite_action"
        summary["root_cause"] = classify_failure(
          reason="nonfinite_action",
          action_finite=False,
          depth_mode=args.depth_mode,
        )
        return False, summary

      env_action = adapter.env_action_from_policy_action(env_policy_action)
      gait_recorder.record_python_step(
        step=step,
        elapsed=elapsed,
        adapter=adapter,
        contract=contract,
        depth=depth,
        raw_policy_action=raw_policy_action,
        applied_policy_action=applied_policy_action,
        env_action=env_action,
        latest_policy_diag=latest_policy_diag,
        latest_frame_diag=latest_frame_diag,
        latest_route_diag=latest_route_diag,
        fall_signals=adapter.fall_signals(),
      )
      _, _, terminated, timed_out, _ = env.step(
        torch.tensor(env_action, dtype=torch.float32, device=env.device)
      )
      reset_now = bool(torch.as_tensor(terminated).detach().cpu().numpy()[0]) or bool(
        torch.as_tensor(timed_out).detach().cpu().numpy()[0]
      )
      if reset_now:
        reset_count += 1
      last_action = applied_policy_action.copy()
      latest_route_diag = _apply_play_command(
        adapter,
        route_follower=route_follower,
        fixed_command=fixed_command,
      )
      latest_frame_diag = adapter.append_current(last_policy_action=last_action)
      final_fall = adapter.fall_signals()
      distance = float(final_fall["base_pos"][0]) - start_x
      elapsed = (step + 1) * env.step_dt
      video_recorder.capture(elapsed)

      if args.debug_parkour and (step == 0 or (step + 1) % 50 == 0):
        print(
          "[parkour] step="
          f"{step + 1} elapsed={elapsed:.2f}s dx={distance:.3f} "
          f"height={final_fall['base_height']:.3f} "
          f"raw_gravity_z={final_fall['raw_projected_gravity_z']:.3f} "
          "depth_stats=" + json.dumps(depth_provider.diagnostics()["stats"]) + " "
          "action_stats=" + json.dumps(vector_stats(applied_policy_action))
        )

      independent_fall = (
        final_fall["base_height"] < args.fall_height
        or final_fall["raw_projected_gravity_z"] > args.bad_gravity_z
      )
      if reset_now or independent_fall:
        summary.update(
          {
            "status": "failed",
            "failure_reason": "reset_or_fall",
            "root_cause": classify_failure(
              reason="zero_action_fall" if args.agent == "zero" else "reset_or_fall",
              fall_signals=final_fall,
              reset_count=reset_count,
              depth_mode=args.depth_mode,
            ),
            "step": step + 1,
            "elapsed_seconds": elapsed,
            "distance_x": distance,
            "reset_count": reset_count,
            "fall_signals": final_fall,
            "latest_frame": latest_frame_diag.__dict__,
            "latest_policy": latest_policy_diag,
            "latest_route": latest_route_diag,
            "depth": depth_provider.diagnostics(),
            "video": video_recorder.diagnostics(),
            "gait_record": gait_recorder.diagnostics(),
          }
        )
        return False, summary

      if distance >= walk_distance:
        break

    depth_diag = depth_provider.diagnostics()
    depth_stats = depth_diag.get("stats", {})
    depth_contract_met = (
      depth_diag.get("shape") == list(contract.depth_shape)
      and depth_diag.get("size") == contract.depth_size
      and all(np.isfinite(float(depth_stats.get(key, 0.0))) for key in ("min", "max", "mean"))
    )
    traversal_accepted = distance >= args.walk_distance
    accepted = distance >= args.walk_distance
    if args.depth_contract_only and depth_contract_met:
      accepted = True
    summary.update(
      {
        "status": "ok" if accepted else "failed",
        "step": min(max_steps, int(round(elapsed / env.step_dt))) if elapsed else 0,
        "elapsed_seconds": elapsed,
        "distance_x": distance,
        "reset_count": reset_count,
        "fall_signals": final_fall,
        "latest_frame": latest_frame_diag.__dict__ if latest_frame_diag is not None else None,
        "latest_policy": latest_policy_diag,
        "latest_route": latest_route_diag,
        "depth": depth_diag,
        "video": video_recorder.diagnostics(),
        "gait_record": gait_recorder.diagnostics(),
        "acceptance": {
          "distance_target_met": traversal_accepted,
          "duration_target_met": elapsed >= args.max_seconds,
          "depth_contract_met": depth_contract_met,
          "depth_contract_only": args.depth_contract_only,
          "no_uncontrolled_fall": True,
          "no_invalid_reset": reset_count == 0,
        },
      }
    )
    if not accepted:
      summary["failure_reason"] = (
        "depth_contract_not_met" if args.depth_contract_only else "distance_target_not_met"
      )
      summary["root_cause"] = classify_failure(
        reason=summary["failure_reason"],
        fall_signals=final_fall,
        reset_count=reset_count,
        depth_mode=args.depth_mode,
      )
    return accepted, summary
  finally:
    gait_recorder.close()
    video_recorder.close()
    depth_viewer.close()
    if hasattr(depth_provider, "close"):
      depth_provider.close()
    env.close()


def run_parkour_play(args: argparse.Namespace) -> int:
  payload: dict[str, Any]
  if args.check_contract:
    payload = run_check_contract(args)
    _write_diagnostics(args.diagnostic_json, payload)
    return 0
  if args.smoke_step:
    payload = run_smoke_step(args)
    _write_diagnostics(args.diagnostic_json, payload)
    return 0
  if args.viewer != "none":
    accepted, payload = run_native_viewer(args)
    _print_json(payload)
    _write_diagnostics(args.diagnostic_json, payload)
    return 0 if accepted else 1
  accepted, payload = run_validate_walk(args)
  _print_json(payload)
  _write_diagnostics(args.diagnostic_json, payload)
  return 0 if accepted else 1


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  raw_args = list(sys.argv[1:] if argv is None else argv)
  args = parser.parse_args(argv)
  validate_explicit = "--validate-walk" in raw_args
  viewer_explicit = "--viewer" in raw_args
  depth_viewer_explicit = "--depth-viewer" in raw_args or "--no-depth-viewer" in raw_args
  if validate_explicit and not viewer_explicit:
    args.viewer = "none"
  if validate_explicit and not depth_viewer_explicit:
    args.depth_viewer = False
  if not (args.check_contract or args.smoke_step or args.validate_walk):
    args.validate_walk = True
  try:
    return run_parkour_play(args)
  except (FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as exc:
    parser.exit(status=2, message=f"error: {exc}\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
