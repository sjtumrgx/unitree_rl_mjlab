from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "configure_topology_getup_depth_topic.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_configure_topology_getup_depth_topic_{len(sys.modules)}"
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


def test_default_deploy_yaml_points_to_g1_getup_tree() -> None:
  module = _load_module()
  assert module.DEFAULT_DEPLOY_YAML.as_posix().endswith(
    "deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml"
  )


def test_update_depth_topic_patches_yaml_fields(tmp_path: Path) -> None:
  module = _load_module()
  deploy_yaml = tmp_path / "deploy.yaml"
  deploy_yaml.write_text(
    yaml.safe_dump(
      {
        "support_geometry_interface": {
          "depth_camera": {
            "topic_name": "",
            "pointcloud_mode": "euclidean_norm",
            "cutoff_distance": 1.5,
            "timeout_ms": 500,
            "retain_last_valid_frame": True,
            "organized_pointcloud": True,
            "pointcloud_field_names": {"x": "x", "y": "y", "z": "z"},
          }
        }
      },
      sort_keys=False,
    )
  )
  updated = module.update_depth_topic(
    deploy_yaml=deploy_yaml,
    topic_name="/g1/depth/points",
    pointcloud_mode="z_depth",
    cutoff_distance=2.0,
    timeout_ms=250,
    retain_last_valid_frame=False,
    x_field_name="px",
    y_field_name="py",
    z_field_name="pz",
  )
  payload = yaml.safe_load(updated.read_text())
  depth_camera = payload["support_geometry_interface"]["depth_camera"]
  assert depth_camera["topic_name"] == "/g1/depth/points"
  assert depth_camera["pointcloud_mode"] == "z_depth"
  assert depth_camera["cutoff_distance"] == 2.0
  assert depth_camera["timeout_ms"] == 250
  assert depth_camera["retain_last_valid_frame"] is False
  assert depth_camera["pointcloud_field_names"] == {"x": "px", "y": "py", "z": "pz"}


def test_main_updates_file_from_cli_args(tmp_path: Path) -> None:
  module = _load_module()
  deploy_yaml = tmp_path / "deploy.yaml"
  deploy_yaml.write_text(
    yaml.safe_dump(
      {
        "support_geometry_interface": {
          "depth_camera": {
            "topic_name": "",
            "pointcloud_mode": "euclidean_norm",
            "cutoff_distance": 1.5,
            "timeout_ms": 500,
            "retain_last_valid_frame": True,
            "organized_pointcloud": True,
            "pointcloud_field_names": {"x": "x", "y": "y", "z": "z"},
          }
        }
      },
      sort_keys=False,
    )
  )
  result = _invoke_main(
    module,
    [
      "--deploy-yaml",
      str(deploy_yaml),
      "--topic-name",
      "/robot/depth/points",
      "--pointcloud-mode",
      "z_depth",
      "--timeout-ms",
      "750",
      "--no-retain-last-valid-frame",
      "--x-field-name",
      "cam_x",
      "--y-field-name",
      "cam_y",
      "--z-field-name",
      "cam_z",
    ],
  )
  assert result == 0
  payload = yaml.safe_load(deploy_yaml.read_text())
  assert payload["support_geometry_interface"]["depth_camera"]["topic_name"] == "/robot/depth/points"
  assert payload["support_geometry_interface"]["depth_camera"]["pointcloud_mode"] == "z_depth"
  assert payload["support_geometry_interface"]["depth_camera"]["timeout_ms"] == 750
  assert payload["support_geometry_interface"]["depth_camera"]["retain_last_valid_frame"] is False
  assert payload["support_geometry_interface"]["depth_camera"]["pointcloud_field_names"] == {
    "x": "cam_x",
    "y": "cam_y",
    "z": "cam_z",
  }
