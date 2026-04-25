from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg

from src.assets.robots.unitree_g1.g1_constants import get_g1_parkour_obstacle_debug_spec
from src.parkour.contract import ACTION_SIZE, assert_no_stale_sensor_references
from src.tasks.velocity.config.g1_parkour.env_cfgs import (
  PARKOUR_OBSTACLE_DEBUG_GEOMS,
  PARKOUR_OBSTACLE_DEBUG_TASK_ID,
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
