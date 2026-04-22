from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "inspect_topology_getup_depth_capture.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_inspect_topology_getup_depth_capture_{len(sys.modules)}"
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


def _write_deploy_yaml(path: Path, *, patch_shape: tuple[int, int] = (4, 4)) -> None:
  path.write_text(
    yaml.safe_dump(
      {
        "support_geometry_interface": {
          "patch_shape": list(patch_shape),
          "depth_camera": {"sensor_name": "support_depth"},
        }
      },
      sort_keys=False,
    )
  )


def test_summarize_depth_capture_reports_stats(tmp_path: Path) -> None:
  module = _load_module()
  deploy_yaml = tmp_path / "deploy.yaml"
  _write_deploy_yaml(deploy_yaml)
  capture_npz = tmp_path / "capture.npz"
  frames = np.array(
    [
      np.linspace(0.0, 1.0, 16, dtype=np.float32),
      np.linspace(0.1, 1.1, 16, dtype=np.float32),
    ]
  )
  np.savez(capture_npz, support_depth=frames)

  summary = module.summarize_depth_capture(capture_npz=capture_npz, deploy_yaml=deploy_yaml)

  assert summary["num_frames"] == 2
  assert summary["patch_shape"] == [4, 4]
  assert summary["frame_size"] == 16
  assert summary["min_value"] == pytest.approx(0.0)
  assert summary["max_value"] == pytest.approx(1.1)
  assert summary["zero_fraction"] > 0.0


def test_main_writes_summary_and_png_artifacts(tmp_path: Path) -> None:
  module = _load_module()
  deploy_yaml = tmp_path / "deploy.yaml"
  _write_deploy_yaml(deploy_yaml)
  capture_npz = tmp_path / "capture.npz"
  np.savez(
    capture_npz,
    support_depth=np.array(
      [
        np.linspace(0.0, 1.0, 16, dtype=np.float32),
        np.linspace(0.2, 0.8, 16, dtype=np.float32),
      ]
    ),
  )
  output_json = tmp_path / "summary.json"
  output_dir = tmp_path / "artifacts"

  result = _invoke_main(
    module,
    [
      str(capture_npz),
      "--deploy-yaml",
      str(deploy_yaml),
      "--output",
      str(output_json),
      "--artifact-dir",
      str(output_dir),
    ],
  )

  assert result == 0
  summary = json.loads(output_json.read_text())
  assert summary["patch_shape"] == [4, 4]
  assert (output_dir / "first_frame.png").exists()
  assert (output_dir / "last_frame.png").exists()
  assert (output_dir / "mean_frame.png").exists()
