from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

DEFAULT_TASK = "Unitree-G1-Parkour-FlatDebug"
DEFAULT_POLICY_DIR = Path("deploy/robots/g1_parkour/config/policy/parkour/v0")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Run the exported depth-conditioned Unitree G1 parkour ONNX policy in a "
      "MuJoCo flat-debug harness. Constant depth is the default first-stage "
      "ablation so proprio/action/asset alignment can be debugged separately."
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
    default="constant",
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
  parser.add_argument("--max-seconds", type=float, default=20.0, help="Validation duration.")
  parser.add_argument("--max-steps", type=int, help="Override validation step count.")
  parser.add_argument("--walk-distance", type=float, default=5.0, help="Forward displacement acceptance target.")
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
    default="none",
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
  parser.add_argument("--diagnostic-json", type=Path, help="Write diagnostics summary JSON to this path.")
  parser.add_argument(
    "--depth-debug-dir",
    type=Path,
    help=(
      "Optional directory for renderer-depth previews/stat artifacts. "
      "Currently used by --depth-mode mujoco."
    ),
  )
  parser.add_argument(
    "--depth-viewer",
    action="store_true",
    help=(
      "Open a live grayscale window for the current normalized depth image. "
      "Use with --depth-mode mujoco to inspect the real parkour_depth_camera stream."
    ),
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
  if args.depth_mode == "mujoco" and not os.environ.get("MUJOCO_GL") and not (
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
    args: argparse.Namespace,
  ) -> None:
    self.env = env
    self.contract = contract
    self.policy = policy
    self.depth_provider = depth_provider
    self.depth_viewer = depth_viewer
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
    self.adapter.set_fixed_command()
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
      self.adapter.set_fixed_command()
      self.latest_frame_diag = self.adapter.append_current(last_policy_action=self.last_action)
      self.latest_fall_signals = self.adapter.fall_signals()
      self.distance_x = float(self.latest_fall_signals["base_pos"][0]) - self.start_x

    self.adapter.set_fixed_command()
    proprio = self.adapter.proprio()
    depth = self.depth_provider.stack(self.adapter)
    assert_depth_contract(depth)
    if self.depth_viewer is not None:
      self.depth_viewer.update(
        _depth_display_frame(self.depth_provider, depth, self.args.depth_viewer_frame),
        self.depth_provider.diagnostics(),
      )

    if self.policy is None:
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
  viewer_env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
  viewer_policy = ParkourNativeViewerPolicy(
    env=raw_env,
    contract=contract,
    policy=policy,
    depth_provider=depth_provider,
    depth_viewer=depth_viewer,
    args=args,
  )
  raw_env.reset()
  viewer_policy.reset()
  num_steps = None
  if not args.viewer_run_until_closed:
    num_steps = args.max_steps or max(1, int(round(args.max_seconds / raw_env.step_dt)))
  print(
    "[parkour] Launching native MuJoCo viewer. "
    "Close the window to stop; use --viewer-run-until-closed for no max-seconds cap."
  )
  try:
    NativeMujocoViewer(
      viewer_env,
      viewer_policy,
      frame_rate=args.viewer_frame_rate,
      enable_perturbations=True,
    ).run(num_steps=num_steps)
    payload = viewer_policy.diagnostics()
    payload.update(
      {
        "policy_dir": paths.policy_dir,
        "exported_dir": paths.exported_dir,
        "onnx": policy.metadata.as_dict() if policy is not None else None,
        "viewer": args.viewer,
        "viewer_frame_rate": args.viewer_frame_rate,
        "viewer_run_until_closed": args.viewer_run_until_closed,
      }
    )
    return True, payload
  finally:
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

  summary: dict[str, Any] = {
    "mode": "validate-walk",
    "task": args.task,
    "agent": args.agent,
    "policy_dir": paths.policy_dir,
    "exported_dir": paths.exported_dir,
    "depth": depth_provider.diagnostics(),
    "onnx": policy.metadata.as_dict() if policy is not None else None,
    "command": [args.command_x, args.command_y, args.command_yaw],
    "policy_frame": args.policy_frame,
    "joint_order": args.joint_order,
    "action_order": args.action_order,
    "max_seconds": args.max_seconds,
    "walk_distance": args.walk_distance,
    "depth_contract_only": args.depth_contract_only,
    "startup_blend_seconds": args.startup_blend_seconds,
    "action_clip": args.action_clip,
    "action_gain": args.action_gain,
    "action_delay_steps": args.action_delay_steps,
  }
  reset_count = 0
  latest_frame_diag: Any = None
  latest_policy_diag: Mapping[str, Any] = {}
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
    adapter.set_fixed_command()
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

    max_steps = args.max_steps or max(1, int(round(args.max_seconds / env.step_dt)))
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
      adapter.set_fixed_command()
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
      _, _, terminated, timed_out, _ = env.step(
        torch.tensor(env_action, dtype=torch.float32, device=env.device)
      )
      reset_now = bool(torch.as_tensor(terminated).detach().cpu().numpy()[0]) or bool(
        torch.as_tensor(timed_out).detach().cpu().numpy()[0]
      )
      if reset_now:
        reset_count += 1
      last_action = applied_policy_action.copy()
      adapter.set_fixed_command()
      latest_frame_diag = adapter.append_current(last_policy_action=last_action)
      final_fall = adapter.fall_signals()
      distance = float(final_fall["base_pos"][0]) - start_x
      elapsed = (step + 1) * env.step_dt

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
            "depth": depth_provider.diagnostics(),
          }
        )
        return False, summary

      if distance >= args.walk_distance:
        break

    depth_diag = depth_provider.diagnostics()
    depth_stats = depth_diag.get("stats", {})
    depth_contract_met = (
      depth_diag.get("shape") == list(contract.depth_shape)
      and depth_diag.get("size") == contract.depth_size
      and all(np.isfinite(float(depth_stats.get(key, 0.0))) for key in ("min", "max", "mean"))
    )
    traversal_accepted = distance >= args.walk_distance
    accepted = traversal_accepted or (args.depth_contract_only and depth_contract_met)
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
        "depth": depth_diag,
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
  args = parser.parse_args(argv)
  if not (args.check_contract or args.smoke_step or args.validate_walk):
    args.validate_walk = True
  try:
    return run_parkour_play(args)
  except (FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as exc:
    parser.exit(status=2, message=f"error: {exc}\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
