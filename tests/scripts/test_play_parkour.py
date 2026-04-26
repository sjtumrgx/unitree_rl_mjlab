from __future__ import annotations

import importlib.util
import inspect
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "play_parkour.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_play_parkour_runtime_{len(sys.modules)}"
  sys.modules.pop(unique_name, None)
  spec = importlib.util.spec_from_file_location(unique_name, SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _invoke_main(module, args: list[str]):
  main = getattr(module, "main")
  parameters = inspect.signature(main).parameters
  if len(parameters) == 0:
    old_argv = sys.argv
    sys.argv = [str(SCRIPT_PATH), *args]
    try:
      return main()
    finally:
      sys.argv = old_argv
  return main(args)


def test_help_documents_parkour_contract_and_diagnostic_flags() -> None:
  module = _load_module()
  help_text = module.build_parser().format_help()

  for flag in (
    "--policy-dir",
    "--exported-dir",
    "--depth-mode",
    "--constant-depth",
    "--viewer",
    "--viewer-frame-rate",
    "--viewer-run-until-closed",
    "--check-contract",
    "--smoke-step",
    "--validate-walk",
    "--depth-contract-only",
    "--diagnostic-json",
    "--depth-debug-dir",
    "--depth-viewer",
    "--depth-viewer-frame",
    "--depth-viewer-frame-rate",
    "--policy-frame",
    "--startup-blend-seconds",
    "--no-depth-viewer",
    "--command-mode",
    "--terrain-route-speed",
    "--terrain-route-lookahead",
    "--video",
    "--video-dir",
    "--gait-record-jsonl",
    "--gait-record-every",
  ):
    assert flag in help_text
  assert "Unitree-G1-Parkour" in help_text
  assert module.DEFAULT_TASK == "Unitree-G1-Parkour"


def test_parser_defaults_use_mujoco_viewer_policy_depth_and_route_command() -> None:
  module = _load_module()
  args = module.build_parser().parse_args([])

  assert args.task == "Unitree-G1-Parkour"
  assert args.depth_mode == "mujoco"
  assert args.joint_order == "isaac"
  assert args.action_order == "isaac"
  assert args.viewer == "native"
  assert args.depth_viewer is True
  assert args.depth_viewer_frame == "policy"
  assert args.command_mode == "terrain-route"
  assert args.terrain_route_speed == 0.25
  assert args.walk_distance is None
  assert args.max_seconds is None
  assert args.video is False
  assert args.video_width == 1920
  assert args.video_height == 1080
  assert args.gait_record_jsonl is None
  assert args.gait_record_every == 1


def test_parser_keeps_headless_debug_overrides_available() -> None:
  module = _load_module()
  args = module.build_parser().parse_args(
    [
      "--viewer",
      "none",
      "--no-depth-viewer",
      "--depth-mode",
      "constant",
      "--command-mode",
      "fixed",
    ]
  )

  assert args.viewer == "none"
  assert args.depth_viewer is False
  assert args.depth_mode == "constant"
  assert args.command_mode == "fixed"


def test_explicit_validate_walk_runs_headless_unless_viewer_requested(monkeypatch) -> None:
  module = _load_module()
  captured = {}

  def fake_run(args):
    captured["args"] = args
    return 0

  monkeypatch.setattr(module, "run_parkour_play", fake_run)

  assert _invoke_main(module, ["--validate-walk"]) == 0

  args = captured["args"]
  assert args.viewer == "none"
  assert args.depth_viewer is False


def test_terrain_route_command_steers_back_to_centerline() -> None:
  module = _load_module()
  follower = module.ParkourTerrainRouteFollower(
    waypoints=((0.0, 0.0), (2.0, 0.0), (4.0, 0.0)),
    speed=0.3,
    lookahead=1.0,
    max_lateral_speed=0.35,
    max_yaw_rate=0.8,
    yaw_gain=1.5,
  )

  command, diagnostics = follower.command(
    base_pos=(1.0, 0.45, 0.8),
    root_quat=(1.0, 0.0, 0.0, 0.0),
  )

  assert command[0] > 0.0
  assert command[1] < 0.0
  assert command[2] < 0.0
  assert diagnostics["target_waypoint"] == [2.0, 0.0]


def test_terrain_route_command_stops_after_final_waypoint() -> None:
  module = _load_module()
  follower = module.ParkourTerrainRouteFollower(
    waypoints=((0.0, 0.0), (2.0, 0.0), (4.0, 0.0)),
    speed=0.3,
    lookahead=1.0,
    max_lateral_speed=0.35,
    max_yaw_rate=0.8,
    yaw_gain=1.5,
  )

  command, diagnostics = follower.command(
    base_pos=(4.01, 0.05, 0.8),
    root_quat=(1.0, 0.0, 0.0, 0.0),
  )

  assert command == (0.0, 0.0, 0.0)
  assert diagnostics["route_completed"] is True


def test_default_play_targets_use_route_endpoint_and_enough_time() -> None:
  module = _load_module()
  env = SimpleNamespace(
    cfg=SimpleNamespace(g1_parkour_route_waypoints=((0.0, 0.0), (25.2, 0.0))),
  )
  args = module.build_parser().parse_args([])

  assert module._resolve_walk_distance(args, env) == 25.2
  assert module._resolve_max_seconds(args, env) > 100.0


def test_terrain_route_speed_controls_route_timing_without_changing_fixed_command() -> None:
  module = _load_module()
  env = SimpleNamespace(
    cfg=SimpleNamespace(g1_parkour_route_waypoints=((0.0, 0.0), (25.2, 0.0))),
  )
  args = module.build_parser().parse_args(["--terrain-route-speed", "0.5"])

  assert args.command_x == 0.25
  assert args.terrain_route_speed == 0.5
  assert module._resolve_max_seconds(args, env) == 25.2 / 0.5 + 10.0


def test_video_defaults_to_exported_model_directory_and_1080p() -> None:
  module = _load_module()
  args = module.build_parser().parse_args(["--video"])
  paths = SimpleNamespace(exported_dir=Path("/tmp/policy/exported"))

  output = module._resolve_video_output_path(args, paths)

  assert output.parent == Path("/tmp/policy/exported")
  assert output.suffix == ".mp4"
  assert args.video_width == 1920
  assert args.video_height == 1080


def test_video_dir_override_is_treated_as_output_directory() -> None:
  module = _load_module()
  args = module.build_parser().parse_args(["--video", "--video-dir", "/tmp/parkour-videos"])
  paths = SimpleNamespace(exported_dir=Path("/tmp/policy/exported"))

  output = module._resolve_video_output_path(args, paths)

  assert output.parent == Path("/tmp/parkour-videos")
  assert output.suffix == ".mp4"


def test_gait_recorder_diagnostics_and_decimation(tmp_path) -> None:
  module = _load_module()
  path = tmp_path / "gait.jsonl"
  recorder = module.GaitJsonlRecorder(path=path, every=3, source="python_play")

  assert recorder.enabled is True
  assert recorder.diagnostics() == {
    "path": str(path),
    "every": 3,
    "source": "python_play",
    "samples": 0,
  }


def test_native_viewer_requires_graphical_display(monkeypatch) -> None:
  module = _load_module()
  monkeypatch.delenv("DISPLAY", raising=False)
  monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

  try:
    module._require_graphical_display()
  except RuntimeError as exc:
    assert "DISPLAY or WAYLAND_DISPLAY" in str(exc)
  else:  # pragma: no cover - this branch is the failure condition.
    raise AssertionError("native viewer should require a graphical display")


def test_native_viewer_uses_rsl_rl_wrapper_for_observations() -> None:
  module = _load_module()
  source = inspect.getsource(module.run_native_viewer)

  assert "RslRlVecEnvWrapper" in source


def test_constant_depth_contract_shape_and_clipping() -> None:
  from src.parkour.contract import DEPTH_SHAPE, DEPTH_SIZE, constant_depth_stack

  depth = constant_depth_stack(1.5)

  assert depth.shape == DEPTH_SHAPE
  assert depth.size == DEPTH_SIZE
  assert depth.dtype == np.float32
  assert float(depth.min()) == 1.0
  assert float(depth.max()) == 1.0


def test_mujoco_depth_mode_is_documented_as_renderer_depth() -> None:
  module = _load_module()
  help_text = module.build_parser().format_help()

  assert "parkour_depth_camera" in help_text
  assert "stage 2" not in help_text.lower()


def test_native_viewer_resets_depth_provider_after_env_reset() -> None:
  module = _load_module()
  source = inspect.getsource(module.run_native_viewer)

  assert "raw_env.reset()" in source
  assert "viewer_policy.reset()" in source
  assert "debug_dir=args.depth_debug_dir" in source


def test_depth_viewer_is_wired_into_native_and_validation_loops() -> None:
  module = _load_module()
  native_source = inspect.getsource(module.run_native_viewer)
  validate_source = inspect.getsource(module.run_validate_walk)
  policy_source = inspect.getsource(module.ParkourNativeViewerPolicy.__call__)

  assert "LiveDepthViewer(" in native_source
  assert "LiveDepthViewer(" in validate_source
  assert "depth_viewer.update" in validate_source
  assert "depth_viewer.update" in policy_source
  assert "_depth_display_frame" in validate_source
  assert "_depth_display_frame" in policy_source


def test_depth_viewer_requires_display_for_headless_validation(monkeypatch) -> None:
  module = _load_module()
  monkeypatch.delenv("DISPLAY", raising=False)
  monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
  args = module.build_parser().parse_args(["--validate-walk", "--depth-viewer"])

  try:
    module.run_validate_walk(args)
  except RuntimeError as exc:
    assert "DISPLAY or WAYLAND_DISPLAY" in str(exc)
  else:  # pragma: no cover - this branch is the failure condition.
    raise AssertionError("depth viewer should require a graphical display")
