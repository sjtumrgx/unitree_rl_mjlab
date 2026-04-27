from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG_PATH = ROOT / 'simulate' / 'config_parkour.yaml'
PARAM_PATH = ROOT / 'simulate' / 'src' / 'param.h'
MAIN_CPP = ROOT / 'simulate' / 'src' / 'main.cc'
DEPTH_BRIDGE_HEADER = ROOT / 'simulate' / 'src' / 'parkour_depth_bridge.h'
DEPTH_BRIDGE_CPP = ROOT / 'simulate' / 'src' / 'parkour_depth_bridge.cc'


def test_parkour_sim_config_preserves_near_pixels_for_policy_depth_parity() -> None:
  payload = yaml.safe_load(SIM_CONFIG_PATH.read_text())
  assert payload['depth_camera_min_distance'] == 0.0


def test_sim_param_struct_loads_depth_camera_min_distance() -> None:
  text = PARAM_PATH.read_text()
  assert 'float depth_camera_min_distance = 0.0f;' in text
  assert 'cfg["depth_camera_min_distance"]' in text


def test_depth_bridge_clips_sub_min_distance_hits_before_publish() -> None:
  text = DEPTH_BRIDGE_CPP.read_text()
  assert 'param::config.depth_camera_min_distance' in text
  assert 'value < param::config.depth_camera_min_distance' in text or 'linear_depth[i] < param::config.depth_camera_min_distance' in text


def test_parkour_sim_config_uses_base_camera_alignment_for_play_parity() -> None:
  payload = yaml.safe_load(SIM_CONFIG_PATH.read_text())
  assert payload['depth_camera_ray_alignment'] == 'base'


def test_depth_bridge_waits_for_initialized_data_before_rendering() -> None:
  header = DEPTH_BRIDGE_HEADER.read_text()
  source = DEPTH_BRIDGE_CPP.read_text()
  main = MAIN_CPP.read_text()
  assert 'std::atomic<bool>* data_ready' in header
  assert 'data_ready_ && !data_ready_->load()' in source
  assert source.index('data_ready_ && !data_ready_->load()') < source.index('ensure_render_resources()')
  assert '&mujoco_data_initialized' in main


def test_depth_bridge_repairs_policy_crop_bottom_artifact_before_publish() -> None:
  payload = yaml.safe_load(SIM_CONFIG_PATH.read_text())
  param = PARAM_PATH.read_text()
  header = DEPTH_BRIDGE_HEADER.read_text()
  source = DEPTH_BRIDGE_CPP.read_text()
  assert payload['depth_policy_bottom_artifact_rows'] == 2
  assert 'int depth_policy_bottom_artifact_rows = 0;' in param
  assert 'cfg["depth_policy_bottom_artifact_rows"]' in param
  assert 'repair_policy_crop_bottom_artifact_band' in header
  assert 'repair_policy_crop_bottom_artifact_band(linear_depth, raw_width, raw_height);' in source
  assert source.index('repair_policy_crop_bottom_artifact_band(linear_depth, raw_width, raw_height);') < source.index('publish_pointcloud')
  assert 'last_valid_top_row' in source
