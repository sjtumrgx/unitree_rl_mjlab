from __future__ import annotations

import inspect

import numpy as np

from src.parkour.contract import DEPTH_SHAPE, DEPTH_SIZE, load_depth_interface_contract
from src.tasks.velocity.rl.parkour_play import MujocoRendererDepthProvider, make_depth_provider


def test_depth_interface_contract_loads_canonical_deploy_metadata() -> None:
  contract = load_depth_interface_contract()

  assert contract.camera_name == "parkour_depth_camera"
  assert contract.raw_resolution == (64, 36)
  assert contract.crop_region == (18, 0, 16, 16)
  assert contract.output_resolution == (32, 18)
  assert contract.depth_range == (0.0, 2.5)
  assert contract.output_range == (0.0, 1.0)
  assert contract.history_source_length == 37
  assert contract.history_skip_frames == 5
  assert contract.num_output_frames == 8
  assert contract.depth_shape == DEPTH_SHAPE
  assert contract.expected_size == DEPTH_SIZE


def test_mujoco_provider_normalizes_crops_and_seeds_history() -> None:
  contract = load_depth_interface_contract()
  provider = MujocoRendererDepthProvider(env=object(), contract=contract)
  raw = np.linspace(0.0, 3.0, contract.raw_height * contract.raw_width, dtype=np.float32).reshape(
    contract.raw_height,
    contract.raw_width,
  )

  frame = provider._normalize_and_crop(raw)
  provider._append_source_frame(frame)
  stack = provider._compose_history_stack()

  assert frame.shape == (18, 32)
  assert frame.dtype == np.float32
  assert 0.0 <= float(frame.min()) <= float(frame.max()) <= 1.0
  assert stack.shape == DEPTH_SHAPE
  assert np.allclose(stack[0], frame)
  assert np.allclose(stack[-1], frame)


def test_mujoco_depth_mode_is_backed_by_renderer_provider() -> None:
  provider = make_depth_provider("mujoco", 0.5, env=object())
  assert isinstance(provider, MujocoRendererDepthProvider)

  source = inspect.getsource(MujocoRendererDepthProvider)
  assert "mujoco.Renderer" in source
  assert "enable_depth_rendering" in source
  assert "endswith(\"/\" + self.contract.camera_name)" in source
  assert "history_skip_frames" in source
  assert "visibility_best" in source


def test_play_script_sets_egl_before_importing_renderer_depth_dependencies() -> None:
  import importlib.util
  import sys
  from pathlib import Path

  script_path = Path(__file__).resolve().parents[2] / "scripts" / "play_parkour.py"
  module_name = f"test_play_parkour_egl_bootstrap_{len(sys.modules)}"
  spec = importlib.util.spec_from_file_location(module_name, script_path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  helper_source = inspect.getsource(module._prepare_mujoco_renderer_env)
  validate_source = inspect.getsource(module.run_validate_walk)
  load_env_source = inspect.getsource(module._load_env)

  assert 'args.depth_mode == "mujoco"' in helper_source
  assert 'os.environ["MUJOCO_GL"] = "egl"' in helper_source
  assert validate_source.index('_prepare_mujoco_renderer_env(args)') < validate_source.index('from src.parkour.contract')
  assert load_env_source.index('_prepare_mujoco_renderer_env(args)') < load_env_source.index('from mjlab.envs import ManagerBasedRlEnv')
