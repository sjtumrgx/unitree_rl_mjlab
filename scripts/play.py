"""Script to play RL agent with RSL-RL."""

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

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
from mjlab.viewer.native.keys import KEY_I, KEY_J, KEY_K, KEY_L


def _suppress_external_wrench_visuals(target_data) -> None:
  """Hide GPU-side xfrc overlays in the native viewer without changing physics."""
  target_data.xfrc_applied[:] = 0.0


def _configure_keyboard_impulse_viewer_visuals(viewer) -> None:
  """Disable native-viewer overlays that show keyboard push artifacts."""
  import mujoco

  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 0


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


class _KeyboardImpulseNativeViewer(NativeMujocoViewer):
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
  video_length: int = 200
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
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

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

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    if cfg.keyboard_impulse:
      viewer = _KeyboardImpulseNativeViewer(
        env,
        policy,
        _KeyboardImpulseController(
          env,
          magnitude_n=cfg.keyboard_impulse_magnitude,
          duration_s=cfg.keyboard_impulse_duration_s,
        ),
      )
    else:
      viewer = NativeMujocoViewer(env, policy)
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
