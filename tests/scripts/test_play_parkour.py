from __future__ import annotations

import importlib.util
import inspect
import sys
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
    "--policy-frame",
    "--startup-blend-seconds",
  ):
    assert flag in help_text
  assert "Unitree-G1-Parkour-FlatDebug" in help_text or module.DEFAULT_TASK == "Unitree-G1-Parkour-FlatDebug"


def test_parser_defaults_use_gray_depth_and_onnx_training_order() -> None:
  module = _load_module()
  args = module.build_parser().parse_args([])

  assert args.constant_depth == 0.5
  assert args.joint_order == "isaac"
  assert args.action_order == "isaac"
  assert args.viewer == "none"


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
