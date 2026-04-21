"""Script to play RL agent with RSL-RL."""

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import KEY_BACKSPACE, KEY_I, KEY_J, KEY_K, KEY_L

_DEFAULT_VIDEO_HEIGHT = 1080
_DEFAULT_VIDEO_WIDTH = 1920


def _suppress_external_wrench_visuals(target_data) -> None:
  """Hide GPU-side xfrc overlays in the native viewer without changing physics."""
  target_data.xfrc_applied[:] = 0.0


def _configure_keyboard_impulse_viewer_visuals(viewer) -> None:
  """Disable native-viewer overlays that show keyboard push artifacts."""
  import mujoco

  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 0


def _resolve_video_dimensions(
  *,
  video_enabled: bool,
  video_height: int | None,
  video_width: int | None,
) -> tuple[int | None, int | None]:
  if not video_enabled:
    return video_height, video_width
  return (
    _DEFAULT_VIDEO_HEIGHT if video_height is None else video_height,
    _DEFAULT_VIDEO_WIDTH if video_width is None else video_width,
  )


def _peak_linear_push_speed_mps(
  velocity_range: dict[str, tuple[float, float]] | None,
) -> float | None:
  if not velocity_range:
    return None
  peak = 0.0
  for axis in ("x", "y", "z"):
    lo, hi = velocity_range.get(axis, (0.0, 0.0))
    peak = max(peak, abs(float(lo)), abs(float(hi)))
  return peak if peak > 0.0 else None


def _resolve_recovery_tuned_push_budget_ns(
  task_id: str,
  env: RslRlVecEnvWrapper,
) -> float | None:
  try:
    train_env_cfg = load_env_cfg(task_id, play=False)
  except Exception:
    return None

  push_event = getattr(train_env_cfg, "events", {}).get("push_robot")
  if push_event is None:
    return None

  peak_speed_mps = _peak_linear_push_speed_mps(
    getattr(push_event, "params", {}).get("velocity_range")
  )
  if peak_speed_mps is None:
    return None

  body_mass = getattr(getattr(env.unwrapped.sim, "model", None), "body_mass", None)
  body_mass_shape = getattr(body_mass, "shape", None)
  if body_mass is None or body_mass_shape is None or body_mass_shape[0] == 0:
    return None

  total_mass_kg = float(body_mass[0].sum().item())
  if total_mass_kg <= 0.0:
    return None

  return total_mass_kg * peak_speed_mps


def _resolve_recovery_tuned_push_limit_n(
  task_id: str,
  env: RslRlVecEnvWrapper,
  duration_s: float,
) -> float | None:
  if duration_s <= 0.0:
    return None
  impulse_budget_ns = _resolve_recovery_tuned_push_budget_ns(task_id, env)
  if impulse_budget_ns is None:
    return None
  return impulse_budget_ns / duration_s


def _clamp_wrench_to_force_limit(
  force: np.ndarray,
  torque: np.ndarray,
  *,
  max_force_n: float | None,
) -> tuple[np.ndarray, np.ndarray]:
  if max_force_n is None or max_force_n <= 0.0:
    return force, torque

  force_mag = float(np.linalg.norm(force))
  if force_mag <= max_force_n or force_mag <= 1e-8:
    return force, torque

  scale = max_force_n / force_mag
  return force * scale, torque * scale


def _clamp_wrench_to_impulse_budget(
  force: np.ndarray,
  torque: np.ndarray,
  *,
  remaining_linear_impulse_ns: float | None,
  step_dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
  if (
    remaining_linear_impulse_ns is None
    or step_dt <= 0.0
    or remaining_linear_impulse_ns == float("inf")
  ):
    return force, torque, float(np.linalg.norm(force)) * max(step_dt, 0.0)

  if remaining_linear_impulse_ns <= 0.0:
    return np.zeros_like(force), np.zeros_like(torque), 0.0

  step_impulse_ns = float(np.linalg.norm(force)) * step_dt
  if step_impulse_ns <= remaining_linear_impulse_ns or step_impulse_ns <= 1e-8:
    return force, torque, step_impulse_ns

  scale = remaining_linear_impulse_ns / step_impulse_ns
  return force * scale, torque * scale, remaining_linear_impulse_ns


def _draw_drag_visual(
  visualizer,
  drag_state: dict[str, object] | None,
  *,
  max_force_n: float | None,
) -> None:
  if not drag_state or not drag_state.get("active", False):
    return

  point = np.asarray(drag_state["point"], dtype=np.float64)
  force = np.asarray(drag_state["force"], dtype=np.float64)
  force_mag = float(np.linalg.norm(force))
  if force_mag <= 1e-8:
    return

  meansize = float(getattr(visualizer, "meansize", 1.0))
  force_frac = min(1.0, force_mag / max_force_n) if max_force_n and max_force_n > 0 else 1.0
  length = meansize * (0.35 + 0.65 * force_frac)
  end = point + (force / force_mag) * length
  visualizer.add_cylinder(
    point,
    end,
    radius=max(0.004, 0.012 * meansize),
    color=(1.0, 0.0, 0.0, 1.0),
    label="play_drag_force",
  )


def _install_drag_video_visualization(
  env: ManagerBasedRlEnv,
  *,
  max_force_n: float | None,
) -> None:
  if not hasattr(env, "update_visualizers"):
    return
  if getattr(env, "_play_drag_visual_hook_installed", False):
    return

  original_update_visualizers = env.update_visualizers

  def wrapped_update_visualizers(visualizer) -> None:
    original_update_visualizers(visualizer)
    drag_state = getattr(env, "_play_drag_visual_state", None)
    _draw_drag_visual(visualizer, drag_state, max_force_n=max_force_n)

  env.update_visualizers = wrapped_update_visualizers  # type: ignore[method-assign]
  env._play_drag_visual_hook_installed = True
  env._play_drag_visual_state = {
    "active": False,
    "point": np.zeros(3, dtype=np.float64),
    "force": np.zeros(3, dtype=np.float64),
  }


class _RecoveryTunedNativeViewer(NativeMujocoViewer):
  """Native viewer with reset shortcut parity and optional perturbation clamping."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    policy,
    *,
    max_perturb_force_n: float | None = None,
    max_linear_impulse_ns: float | None = None,
    key_callback=None,
    enable_perturbations: bool = True,
  ) -> None:
    self._max_perturb_force_n = max_perturb_force_n
    self._max_linear_impulse_ns = max_linear_impulse_ns
    self._remaining_linear_impulse_ns: float | None = None
    self._drag_budget_active = False
    self._last_applied_force_n = 0.0
    super().__init__(
      env,
      policy,
      key_callback=key_callback,
      enable_perturbations=enable_perturbations,
    )

  def setup(self) -> None:
    super().setup()
    print("[INFO] Native viewer shortcuts: Enter/Backspace=reset.")
    if self._max_perturb_force_n is not None and self.enable_perturbations:
      print(
        "[INFO] Current push limit: "
        f"{self._max_perturb_force_n:.1f} N "
        "(training-aligned max external push)."
      )
    if self._max_linear_impulse_ns is not None and self.enable_perturbations:
      print(
        "[INFO] Current drag impulse budget: "
        f"{self._max_linear_impulse_ns:.1f} N·s per continuous drag."
      )

  def _set_status_overlay(self, viewer) -> None:
    import mujoco

    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    remaining_budget = (
      self._remaining_linear_impulse_ns
      if self._remaining_linear_impulse_ns is not None
      else self._max_linear_impulse_ns
    )

    rows = [
      ("Env", f"{self.env_idx + 1}/{self.env.num_envs}"),
      ("Step", f"{status.step_count}"),
      ("Status", f"{'PAUSED' if status.paused else 'RUNNING'}{capped}"),
      ("Speed", f"{status.speed_label}"),
      ("Target RT", f"{status.target_realtime:.2f}x"),
      ("Actual RT", f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)"),
    ]
    if self.enable_perturbations and self._max_perturb_force_n is not None:
      rows.append(("Push limit", f"{self._max_perturb_force_n:.1f} N"))
    if self.enable_perturbations and remaining_budget is not None:
      rows.append(("Drag left", f"{remaining_budget:.1f} N·s"))
    if self.enable_perturbations:
      rows.append(("Push now", f"{self._last_applied_force_n:.1f} N"))

    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      "\n".join(label for label, _ in rows),
      "\n".join(value for _, value in rows),
    )
    viewer.set_texts(overlay)

  def _safe_key_callback(self, key: int) -> None:
    if key == KEY_BACKSPACE:
      self.request_reset()
    super()._safe_key_callback(key)

  def sync_viewer_to_env(self) -> None:
    if self._max_perturb_force_n is None:
      return super().sync_viewer_to_env()

    import mujoco

    v = self.viewer
    if v is None or self.mjm is None or self.mjd is None:
      return

    sim_data = self.env.unwrapped.sim.data
    pert = v.perturb
    perturb_active = pert.active != 0 and pert.select > 0
    drag_force = np.zeros(3, dtype=np.float64)
    drag_point = np.zeros(3, dtype=np.float64)

    if perturb_active and not self._drag_budget_active:
      self._remaining_linear_impulse_ns = self._max_linear_impulse_ns
    self._drag_budget_active = perturb_active

    if perturb_active:
      mujoco.mjv_applyPerturbForce(self.mjm, self.mjd, pert)

      body_id = pert.select
      force = self.mjd.xfrc_applied[body_id, :3].copy()
      torque = self.mjd.xfrc_applied[body_id, 3:].copy()
      force, torque = _clamp_wrench_to_force_limit(
        force,
        torque,
        max_force_n=self._max_perturb_force_n,
      )
      force, torque, impulse_used_ns = _clamp_wrench_to_impulse_budget(
        force,
        torque,
        remaining_linear_impulse_ns=self._remaining_linear_impulse_ns,
        step_dt=float(self.env.unwrapped.step_dt),
      )
      self._last_applied_force_n = float(np.linalg.norm(force))
      if self._remaining_linear_impulse_ns is not None:
        self._remaining_linear_impulse_ns = max(
          0.0,
          self._remaining_linear_impulse_ns - impulse_used_ns,
        )
      point = self.mjd.xipos[body_id].copy()
      drag_force = force.copy()
      drag_point = point.copy()

      qfrc = np.zeros(self.mjm.nv)
      mujoco.mj_applyFT(self.mjm, self.mjd, force, torque, point, body_id, qfrc)

      sim_data.qfrc_applied[self.env_idx] = torch.from_numpy(qfrc).to(
        device=sim_data.qfrc_applied.device,
        dtype=sim_data.qfrc_applied.dtype,
      )
      self.mjd.xfrc_applied[body_id] = 0.0
    else:
      self._remaining_linear_impulse_ns = None
      self._last_applied_force_n = 0.0
      sim_data.qfrc_applied[self.env_idx] = 0.0

    setattr(
      self.env.unwrapped,
      "_play_drag_visual_state",
      {
        "active": perturb_active and self._last_applied_force_n > 0.0,
        "point": drag_point,
        "force": drag_force,
      },
    )


class _KeyboardImpulseController:
  """Apply short keyboard-triggered body-frame pushes in native play mode."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    *,
    magnitude_n: float,
    duration_s: float,
  ) -> None:
    if magnitude_n <= 0:
      raise ValueError("`keyboard_impulse_magnitude` must be positive.")
    if duration_s <= 0:
      raise ValueError("`keyboard_impulse_duration_s` must be positive.")

    self._env = env.unwrapped
    self._asset = self._env.scene["robot"]
    self._device = self._env.device
    viewer_cfg = getattr(env, "cfg", self._env.cfg).viewer
    self._body_id = self._resolve_body_id(getattr(viewer_cfg, "body_name", None))
    self._magnitude_n = float(magnitude_n)
    self._duration_steps = max(1, math.ceil(duration_s / self._env.step_dt))
    self._point_offset_local = torch.tensor(
      [[0.0, 0.0, 0.25]], device=self._device, dtype=torch.float32
    )
    self._lock = Lock()
    self._pending: tuple[int, tuple[float, float, float]] | None = None
    self._active_env_idx: int | None = None
    self._active_force_w: torch.Tensor | None = None
    self._active_torque_w: torch.Tensor | None = None
    self._remaining_steps = 0

  def request_impulse(
    self,
    *,
    env_idx: int,
    body_frame_direction: tuple[float, float, float],
  ) -> None:
    with self._lock:
      self._pending = (int(env_idx), tuple(float(v) for v in body_frame_direction))

  def sync(self) -> None:
    if self._active_env_idx is not None and self._remaining_steps <= 0:
      self._clear_active_wrench()

    pending = self._pop_pending()
    if pending is not None:
      env_idx, direction = pending
      if self._active_env_idx is not None:
        self._clear_active_wrench()
      force_w, torque_w = self._make_wrench(env_idx, direction)
      self._active_env_idx = env_idx
      self._active_force_w = force_w
      self._active_torque_w = torque_w
      self._remaining_steps = self._duration_steps

    if self._active_env_idx is None:
      return
    assert self._active_force_w is not None
    assert self._active_torque_w is not None
    self._write_wrench(
      self._active_env_idx,
      self._active_force_w,
      self._active_torque_w,
    )
    self._remaining_steps -= 1

  def _pop_pending(self) -> tuple[int, tuple[float, float, float]] | None:
    with self._lock:
      pending = self._pending
      self._pending = None
      return pending

  def _resolve_body_id(self, body_name: str | None) -> int:
    if body_name and body_name in getattr(self._asset, "body_names", ()):
      body_ids, _ = self._asset.find_bodies(body_name)
      return int(body_ids[0])
    return int(self._asset.indexing.root_body_id)

  def _make_wrench(
    self,
    env_idx: int,
    direction_b: tuple[float, float, float],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    direction = torch.tensor(direction_b, device=self._device, dtype=torch.float32)
    direction = direction / torch.linalg.norm(direction).clamp(min=1e-6)
    force_b = (direction * self._magnitude_n).unsqueeze(0)

    body_quat = self._asset.data.body_com_quat_w[env_idx, self._body_id].unsqueeze(0)
    force_w = quat_apply(body_quat, force_b).reshape(1, 1, 3)
    offset_w = quat_apply(body_quat, self._point_offset_local).reshape(1, 1, 3)
    torque_w = torch.cross(offset_w, force_w, dim=-1)
    return force_w, torque_w

  def _write_wrench(
    self,
    env_idx: int,
    forces: torch.Tensor,
    torques: torch.Tensor,
  ) -> None:
    env_ids = torch.tensor([env_idx], device=self._device, dtype=torch.long)
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=[self._body_id],
    )

  def _clear_active_wrench(self) -> None:
    if self._active_env_idx is None:
      return
    zeros = torch.zeros((1, 1, 3), device=self._device, dtype=torch.float32)
    self._write_wrench(self._active_env_idx, zeros, zeros)
    self._active_env_idx = None
    self._active_force_w = None
    self._active_torque_w = None
    self._remaining_steps = 0


class _KeyboardImpulseNativeViewer(_RecoveryTunedNativeViewer):
  """Native viewer with AntiFall keyboard push shortcuts."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    policy,
    impulse_controller: _KeyboardImpulseController,
  ) -> None:
    self._impulse_controller = impulse_controller
    super().__init__(
      env,
      policy,
      max_perturb_force_n=None,
      max_linear_impulse_ns=None,
      key_callback=self._handle_keyboard_impulse_key,
      enable_perturbations=False,
    )

  def setup(self) -> None:
    super().setup()
    assert self.viewer is not None
    _configure_keyboard_impulse_viewer_visuals(self.viewer)
    print(
      "[INFO] Keyboard pushes enabled: I=forward, K=backward, J=left, L=right "
      "(relative to the robot body frame)."
    )

  def _sync_env_state_to_mjdata(self, target_data, sim_data, env_idx: int) -> None:
    super()._sync_env_state_to_mjdata(target_data, sim_data, env_idx)
    _suppress_external_wrench_visuals(target_data)

  def sync_viewer_to_env(self) -> None:
    super().sync_viewer_to_env()
    self._impulse_controller.sync()

  def _handle_keyboard_impulse_key(self, key: int) -> None:
    direction = {
      KEY_I: (1.0, 0.0, 0.0),
      KEY_J: (0.0, 1.0, 0.0),
      KEY_K: (-1.0, 0.0, 0.0),
      KEY_L: (0.0, -1.0, 0.0),
    }.get(key)
    if direction is None:
      return
    self._impulse_controller.request_impulse(
      env_idx=self.env_idx,
      body_frame_direction=direction,
    )


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  keyboard_impulse: bool = False
  keyboard_impulse_magnitude: float = 300.0
  keyboard_impulse_duration_s: float = 0.15
  video: bool = False
  video_length: int | None = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")
  elif cfg.keyboard_impulse:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled for keyboard impulse play")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  resolved_video_height, resolved_video_width = _resolve_video_dimensions(
    video_enabled=bool(cfg.video),
    video_height=cfg.video_height,
    video_width=cfg.video_width,
  )
  if resolved_video_height is not None:
    env_cfg.viewer.height = resolved_video_height
  if resolved_video_width is not None:
    env_cfg.viewer.width = resolved_video_width

  # Handle "auto" viewer selection before env creation so video hooks can follow it.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  video_drag_force_limit_n: float | None = None
  if cfg.video and resolved_viewer == "native":
    preview_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    video_drag_force_limit_n = _resolve_recovery_tuned_push_limit_n(
      task_id,
      preview_env,
      duration_s=cfg.keyboard_impulse_duration_s,
    )
    _install_drag_video_visualization(env.unwrapped, max_force_n=video_drag_force_limit_n)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  try:
    if resolved_viewer == "native":
      recovery_tuned_push_budget_ns = _resolve_recovery_tuned_push_budget_ns(task_id, env)
      recovery_tuned_push_limit_n = _resolve_recovery_tuned_push_limit_n(
        task_id,
        env,
        duration_s=cfg.keyboard_impulse_duration_s,
      )
      if cfg.keyboard_impulse:
        keyboard_impulse_magnitude_n = cfg.keyboard_impulse_magnitude
        if recovery_tuned_push_limit_n is not None:
          keyboard_impulse_magnitude_n = min(
            keyboard_impulse_magnitude_n,
            recovery_tuned_push_limit_n,
          )
          print(
            "[INFO] Keyboard push cap aligned to training envelope: "
            f"{keyboard_impulse_magnitude_n:.1f} N"
          )
        viewer = _KeyboardImpulseNativeViewer(
          env,
          policy,
          _KeyboardImpulseController(
            env,
            magnitude_n=keyboard_impulse_magnitude_n,
            duration_s=cfg.keyboard_impulse_duration_s,
          ),
        )
      else:
        viewer = _RecoveryTunedNativeViewer(
          env,
          policy,
          max_perturb_force_n=recovery_tuned_push_limit_n,
          max_linear_impulse_ns=recovery_tuned_push_budget_ns,
        )
      viewer.run()
    elif resolved_viewer == "viser":
      if cfg.keyboard_impulse:
        print(
          "[WARN] Keyboard pushes are only available in the native MuJoCo viewer; "
          "continuing without keyboard pushes in viser mode."
        )
      ViserPlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    env.close()


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

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
