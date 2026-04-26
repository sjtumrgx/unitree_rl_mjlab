from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg

from src.assets.robots.unitree_g1.g1_constants import (
  get_g1_parkour_complex_terrain_debug_spec,
)
from src.parkour.contract import ACTION_SIZE, assert_no_stale_sensor_references
from src.tasks.velocity.config.g1_parkour.env_cfgs import (
  PARKOUR_COMPLEX_TERRAIN_GEOMS,
  PARKOUR_TASK_ID,
)

ROOT = Path(__file__).resolve().parents[2]
CXX_COMPLEX_SCENE = (
  ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1_parkour.xml"
)
CXX_SIM_CONFIG = ROOT / "simulate" / "config_parkour.yaml"
CXX_SMOKE_HARNESS = ROOT / "scripts" / "run_g1_parkour_cpp_dds_smoke.py"
CPP_DEPLOY_PARAM = ROOT / "deploy" / "include" / "param.h"
CPP_OBSERVATIONS = (
  ROOT / "deploy" / "include" / "isaaclab" / "envs" / "mdp" / "observations" / "observations.h"
)
CPP_DEPTH_PROVIDER = (
  ROOT / "deploy" / "robots" / "g1_parkour" / "src" / "ParkourDepthProvider.cpp"
)
CPP_DEPTH_BRIDGE = ROOT / "simulate" / "src" / "parkour_depth_bridge.cc"
SIM_PARAM = ROOT / "simulate" / "src" / "param.h"
SIM_MAIN = ROOT / "simulate" / "src" / "main.cc"
DEPLOY_YAML = (
  ROOT / "deploy" / "robots" / "g1_parkour" / "config" / "policy" / "parkour" / "v0" / "params" / "deploy.yaml"
)


def test_only_formal_g1_parkour_task_is_publicly_registered() -> None:
  parkour_tasks = [task for task in list_tasks() if "Parkour" in task]

  assert parkour_tasks == [PARKOUR_TASK_ID]
  assert not any("Debug" in task for task in parkour_tasks)


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
  cfg = load_env_cfg(PARKOUR_TASK_ID, play=True)
  contract = getattr(cfg, "g1_parkour_complex_terrain_contract")

  assert getattr(cfg, "g1_parkour_official") is True
  assert getattr(cfg, "g1_parkour_complex_terrain") is True
  assert getattr(cfg, "g1_parkour_complex_terrain_debug") is False
  assert getattr(cfg, "g1_parkour_flat_debug") is False
  assert getattr(cfg, "g1_parkour_obstacle_debug") is False
  assert (
    getattr(cfg, "g1_parkour_complex_terrain_geoms")
    == PARKOUR_COMPLEX_TERRAIN_GEOMS
  )
  assert contract["target_distance_m"] >= 18.0
  assert contract["up_stairs"] == {
    "steps": 5,
    "step_run_m": 0.36,
    "max_height_m": 0.15,
  }
  assert contract["down_stairs"] == {
    "steps": 5,
    "step_run_m": 0.36,
    "max_height_m": 0.15,
  }
  assert contract["gap"]["keeps_global_floor"] is True
  assert contract["gap"]["lower_strip_width_m"] <= 0.40
  assert contract["gap"]["second_lower_strip_width_m"] <= 0.40
  assert "pyramid_stairs" in contract["instinctlab_reference"][
    "approximated_sub_terrains"
  ]
  assert "square_gaps" in contract["instinctlab_reference"][
    "approximated_sub_terrains"
  ]
  assert len(cfg.actions["joint_pos"].scale) == ACTION_SIZE
  assert_no_stale_sensor_references(cfg)


def _gap_distance(spec, near_name: str, far_name: str) -> float:
  near = next(geom for geom in spec.worldbody.geoms if geom.name == near_name)
  far = next(geom for geom in spec.worldbody.geoms if geom.name == far_name)
  near_far_edge = float(near.pos[0] + near.size[0])
  far_near_edge = float(far.pos[0] - far.size[0])
  return far_near_edge - near_far_edge


def _xml_geom_map(path: Path) -> dict[str, ET.Element]:
  root = ET.parse(path).getroot()
  return {
    str(geom.attrib["name"]): geom
    for geom in root.findall(".//geom")
    if "name" in geom.attrib
  }


def _xml_gap_distance(
  geoms: dict[str, ET.Element],
  near_name: str,
  far_name: str,
) -> float:
  near = geoms[near_name]
  far = geoms[far_name]
  near_pos_x = float(near.attrib["pos"].split()[0])
  near_size_x = float(near.attrib["size"].split()[0])
  far_pos_x = float(far.attrib["pos"].split()[0])
  far_size_x = float(far.attrib["size"].split()[0])
  return (far_pos_x - far_size_x) - (near_pos_x + near_size_x)


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


def test_g1_parkour_complex_terrain_gaps_are_no_more_than_40cm() -> None:
  spec = get_g1_parkour_complex_terrain_debug_spec()

  assert _gap_distance(
    spec,
    "parkour_complex_gap_near_platform",
    "parkour_complex_gap_far_platform",
  ) <= 0.40
  assert _gap_distance(
    spec,
    "parkour_complex_second_gap_near_platform",
    "parkour_complex_second_gap_far_platform",
  ) <= 0.40


def test_cxx_parkour_complex_scene_mirrors_python_play_terrain_assets() -> None:
  geoms = _xml_geom_map(CXX_COMPLEX_SCENE)

  assert "parkour_complex_terrain_course" in CXX_COMPLEX_SCENE.read_text()
  assert {
    "parkour_complex_up_stair_01",
    "parkour_complex_up_stair_05",
    "parkour_complex_top_platform",
    "parkour_complex_down_stair_01",
    "parkour_complex_down_stair_05",
    "parkour_complex_gap_near_platform",
    "parkour_complex_gap_floor_marker",
    "parkour_complex_gap_far_platform",
    "parkour_complex_discrete_box_01",
    "parkour_complex_discrete_box_06",
    "parkour_complex_up_stair_b_04",
    "parkour_complex_second_gap_floor_marker",
    "parkour_complex_mesh_box_01",
    "parkour_complex_mesh_box_06",
  }.issubset(geoms)
  assert (
    float(geoms["parkour_complex_up_stair_05"].attrib["size"].split()[2])
    == 0.075
  )
  assert (
    float(geoms["parkour_complex_up_stair_b_04"].attrib["size"].split()[2])
    == 0.07
  )


def test_cxx_default_interactive_scene_uses_complex_policy_depth_assets() -> None:
  config_text = CXX_SIM_CONFIG.read_text()
  scene_text = CXX_COMPLEX_SCENE.read_text()

  assert "scene_g1_parkour.xml" in config_text
  assert "enable_depth_camera: 1" in config_text
  assert "parkour_depth_camera" in scene_text
  assert "parkour_complex_terrain_course" in scene_text
  assert "parkour_complex_up_stair_05" in scene_text
  assert "parkour_complex_down_stair_05" in scene_text
  assert "parkour_complex_gap_near_platform" in scene_text
  assert "parkour_complex_gap_far_platform" in scene_text
  assert "parkour_complex_discrete_box_06" in scene_text
  assert "parkour_complex_mesh_box_06" in scene_text
  assert "depth_debug_crop_top: 18" in config_text
  assert "depth_debug_crop_left: 16" in config_text
  assert "depth_debug_crop_width: 32" in config_text
  assert "depth_debug_crop_height: 18" in config_text
  assert "depth_debug_policy_display: 1" in config_text
  assert "depth_debug_artifact_floor: 0.10" in config_text
  assert "depth_debug_artifact_ceiling: 0.75" in config_text


def test_cxx_parkour_complex_scene_keeps_gap_spans_at_most_40cm() -> None:
  geoms = _xml_geom_map(CXX_COMPLEX_SCENE)

  assert _xml_gap_distance(
    geoms,
    "parkour_complex_gap_near_platform",
    "parkour_complex_gap_far_platform",
  ) <= 0.40
  assert _xml_gap_distance(
    geoms,
    "parkour_complex_second_gap_near_platform",
    "parkour_complex_second_gap_far_platform",
  ) <= 0.40


def test_cpp_dds_smoke_harness_exposes_complex_terrain_scene_flag() -> None:
  text = CXX_SMOKE_HARNESS.read_text()

  assert "FLAT_SCENE" in text
  assert "args.sim_scene or FLAT_SCENE" in text
  assert "COMPLEX_TERRAIN_SCENE" in text
  assert "--complex-terrain-course" in text
  assert "scene_g1_parkour.xml" in text
  assert (
    "--low-obstacle-course and --complex-terrain-course are mutually exclusive"
    in text
  )


def test_cpp_dds_sim_autostart_defaults_to_live_depth_for_terrain() -> None:
  harness = CXX_SMOKE_HARNESS.read_text()
  param = CPP_DEPLOY_PARAM.read_text()
  provider = CPP_DEPTH_PROVIDER.read_text()

  assert "effective_live_depth_blend = 1.0" in harness
  assert '"--live-depth-blend", str(max(0.0, min(1.0, effective_live_depth_blend)))' in harness
  assert "--depth-artifact-floor" in harness
  assert "--sim-autostart-parkour defaulting to --live-depth-blend=1.0" in param
  assert "param::parkour_live_depth_blend_override" in provider
  assert "param::parkour_constant_depth_override" in provider
  assert "artifact_floor_" in provider


def test_cpp_depth_window_starts_visible_without_dds_ready_gate() -> None:
  bridge = CPP_DEPTH_BRIDGE.read_text()
  sim_param = SIM_PARAM.read_text()

  assert "Parkour Policy Depth" in bridge
  assert "DEPTH_DEBUG_WINDOW_VISIBLE" in bridge
  assert "G1_PARKOUR_DEPTH_DEBUG_WINDOW" in bridge
  assert "while (!stop_requested_ && dds_ready_ && !dds_ready_->load())" not in bridge
  assert "if ((!dds_ready_ || dds_ready_->load()) && !pointcloud_publisher_)" in bridge
  assert "depth_debug_policy_display" in sim_param
  assert "depth_debug_artifact_floor" in sim_param
  assert "depth_debug_artifact_ceiling" in sim_param
  assert "std::clamp(normalized, artifact_floor, artifact_ceiling)" in bridge


def test_cpp_loopback_keyboard_forward_is_hold_to_walk_not_latched() -> None:
  observations = CPP_OBSERVATIONS.read_text()
  deploy_yaml = DEPLOY_YAML.read_text()

  assert "static float parkour_keyboard_cruise_speed" in observations
  assert "static std::vector<float> parkour_keyboard_command" not in observations
  assert "last_forward_key_time" in observations
  assert "hold_timeout_s" in observations
  assert "hold_timeout_s: 0.45" in deploy_yaml
  assert "std::vector<float> parkour_keyboard_command(3, 0.0f);" in observations
  assert "parkour_keyboard_command[0] = idle_speed;" in observations
  assert 'key == "w" || key == "up"' in observations
  assert "parkour_keyboard_command[0] = parkour_keyboard_cruise_speed;" in observations
  assert "key.empty()" in observations
  assert "param::sim_heading_lock" in observations
  assert "param::sim_heading_target_yaw - current_yaw" in observations
  assert "parkour_keyboard_cruise_speed + lin_vel_step" in observations
  assert "parkour_keyboard_cruise_speed - lin_vel_step" in observations


def test_mujoco_reset_uses_configured_parkour_initial_pose() -> None:
  sim_main = SIM_MAIN.read_text()

  assert "PARKOUR_SIM_RESET_REAPPLIED_CONFIGURED_POSE" in sim_main
  assert "d->time + 1.0e-9 < last_observed_sim_time" in sim_main
  assert "apply_configured_pose_to_qpos(m, d->qpos)" in sim_main
  assert "mju_zero(d->qvel, m->nv)" in sim_main
  assert "mju_zero(d->ctrl, m->nu)" in sim_main
  assert "Do not mutate the model default pose" in sim_main
  assert "model->qpos0" not in sim_main
