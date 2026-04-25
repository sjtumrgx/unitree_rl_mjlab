from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg

from src.assets.robots.unitree_g1.g1_constants import (
  get_g1_parkour_complex_terrain_debug_spec,
  get_g1_parkour_obstacle_debug_spec,
)
from src.parkour.contract import ACTION_SIZE, assert_no_stale_sensor_references
from src.tasks.velocity.config.g1_parkour.env_cfgs import (
  PARKOUR_COMPLEX_TERRAIN_DEBUG_GEOMS,
  PARKOUR_COMPLEX_TERRAIN_DEBUG_TASK_ID,
  PARKOUR_OBSTACLE_DEBUG_GEOMS,
  PARKOUR_OBSTACLE_DEBUG_TASK_ID,
  PARKOUR_TASK_ID,
)


def test_g1_parkour_obstacle_debug_task_is_registered() -> None:
  assert PARKOUR_OBSTACLE_DEBUG_TASK_ID in list_tasks()


def test_g1_parkour_obstacle_debug_cfg_marks_conservative_contract() -> None:
  cfg = load_env_cfg(PARKOUR_OBSTACLE_DEBUG_TASK_ID, play=True)

  assert getattr(cfg, "g1_parkour_obstacle_debug") is True
  assert getattr(cfg, "g1_parkour_flat_debug") is False
  assert getattr(cfg, "g1_parkour_obstacle_geoms") == PARKOUR_OBSTACLE_DEBUG_GEOMS
  assert getattr(cfg, "g1_parkour_obstacle_contract") == {
    "low_block_height_m": 0.05,
    "gap_width_m": 0.10,
    "target_distance_m": 3.0,
  }
  assert "robot" in cfg.scene.entities
  assert len(cfg.actions["joint_pos"].scale) == ACTION_SIZE
  assert_no_stale_sensor_references(cfg)


def test_g1_parkour_obstacle_spec_contains_deterministic_low_block_and_gap() -> None:
  spec = get_g1_parkour_obstacle_debug_spec()
  geom_names = {geom.name for geom in spec.worldbody.geoms}

  assert {
    "parkour_debug_low_block",
    "parkour_debug_gap_near_lip",
    "parkour_debug_gap_far_lip",
  }.issubset(geom_names)


def test_g1_parkour_complex_terrain_debug_task_is_registered() -> None:
  assert PARKOUR_COMPLEX_TERRAIN_DEBUG_TASK_ID in list_tasks()


def test_g1_parkour_formal_task_defaults_to_complex_route_terrain() -> None:
  cfg = load_env_cfg(PARKOUR_TASK_ID, play=True)
  route = getattr(cfg, "g1_parkour_route_waypoints")
  contract = getattr(cfg, "g1_parkour_complex_terrain_contract")

  assert PARKOUR_TASK_ID == "Unitree-G1-Parkour"
  assert PARKOUR_TASK_ID in list_tasks()
  assert getattr(cfg, "g1_parkour_official") is True
  assert getattr(cfg, "g1_parkour_complex_terrain") is True
  assert getattr(cfg, "g1_parkour_complex_terrain_debug") is False
  assert len(route) >= 9
  assert route[0] == (0.0, 0.0)
  assert route[-1][0] >= 18.0
  assert contract["target_distance_m"] >= 18.0
  assert len(cfg.actions["joint_pos"].scale) == ACTION_SIZE
  assert_no_stale_sensor_references(cfg)


def test_g1_parkour_complex_terrain_cfg_marks_instinctlab_reference() -> None:
  cfg = load_env_cfg(PARKOUR_COMPLEX_TERRAIN_DEBUG_TASK_ID, play=True)
  contract = getattr(cfg, "g1_parkour_complex_terrain_contract")

  assert getattr(cfg, "g1_parkour_complex_terrain_debug") is True
  assert getattr(cfg, "g1_parkour_flat_debug") is False
  assert getattr(cfg, "g1_parkour_obstacle_debug") is False
  assert (
    getattr(cfg, "g1_parkour_complex_terrain_geoms")
    == PARKOUR_COMPLEX_TERRAIN_DEBUG_GEOMS
  )
  assert contract["target_distance_m"] >= 18.0
  assert contract["up_stairs"] == {
    "steps": 5,
    "step_run_m": 0.36,
    "max_height_m": 0.30,
  }
  assert contract["down_stairs"] == {
    "steps": 5,
    "step_run_m": 0.36,
    "max_height_m": 0.30,
  }
  assert contract["gap"]["keeps_global_floor"] is True
  assert "pyramid_stairs" in contract["instinctlab_reference"][
    "approximated_sub_terrains"
  ]
  assert "square_gaps" in contract["instinctlab_reference"][
    "approximated_sub_terrains"
  ]
  assert len(cfg.actions["joint_pos"].scale) == ACTION_SIZE
  assert_no_stale_sensor_references(cfg)


def test_g1_parkour_complex_terrain_spec_contains_expected_assets() -> None:
  spec = get_g1_parkour_complex_terrain_debug_spec()
  geom_names = {geom.name for geom in spec.worldbody.geoms}

  assert {
    "parkour_complex_up_stair_01",
    "parkour_complex_up_stair_05",
    "parkour_complex_top_platform",
    "parkour_complex_down_stair_01",
    "parkour_complex_down_stair_05",
    "parkour_complex_gap_near_platform",
    "parkour_complex_gap_floor_marker",
    "parkour_complex_gap_far_platform",
    "parkour_complex_up_stair_b_04",
    "parkour_complex_second_gap_floor_marker",
    "parkour_complex_discrete_box_01",
    "parkour_complex_discrete_box_06",
    "parkour_complex_mesh_box_01",
    "parkour_complex_mesh_box_06",
  }.issubset(geom_names)
