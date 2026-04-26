from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG_PATH = ROOT / 'simulate' / 'config_parkour.yaml'
PARAM_PATH = ROOT / 'simulate' / 'src' / 'param.h'
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
