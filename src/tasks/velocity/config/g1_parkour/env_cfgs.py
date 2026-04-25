"""Flat MuJoCo debug task for the exported G1 parkour ONNX policy."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  PARKOUR_COMPLEX_TERRAIN_GEOM_NAMES,
  get_g1_parkour_complex_terrain_robot_cfg,
  get_g1_parkour_obstacle_robot_cfg,
  get_g1_parkour_robot_cfg,
)
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from src.parkour.contract import (
  PARKOUR_SCENE_SENSOR_REMAP,
  TRAINING_JOINT_NAMES,
  assert_no_stale_sensor_references,
  load_deploy_contract,
)

PARKOUR_TASK_ID = "Unitree-G1-Parkour"
PARKOUR_FLAT_DEBUG_TASK_ID = "Unitree-G1-Parkour-FlatDebug"
PARKOUR_OBSTACLE_DEBUG_TASK_ID = "Unitree-G1-Parkour-ObstacleDebug"
PARKOUR_COMPLEX_TERRAIN_DEBUG_TASK_ID = "Unitree-G1-Parkour-ComplexTerrainDebug"
DEFAULT_COMMAND_X = 0.25
PARKOUR_OBSTACLE_DEBUG_GEOMS = (
  "robot/parkour_debug_low_block",
  "robot/parkour_debug_gap_near_lip",
  "robot/parkour_debug_gap_far_lip",
)
PARKOUR_COMPLEX_TERRAIN_DEBUG_GEOMS = tuple(
  f"robot/{name}" for name in PARKOUR_COMPLEX_TERRAIN_GEOM_NAMES
)
PARKOUR_COMPLEX_TERRAIN_ROUTE_WAYPOINTS = (
  (0.0, 0.0),
  (2.0, 0.0),
  (4.8, 0.0),
  (7.45, 0.0),
  (10.8, 0.0),
  (13.2, 0.0),
  (15.5, 0.0),
  (17.8, 0.0),
  (19.75, 0.0),
  (22.0, 0.0),
  (25.2, 0.0),
)
PARKOUR_COMPLEX_TERRAIN_INSTINCTLAB_REFERENCE = {
  "source": (
    "/home/eilab/instinctlab/source/instinctlab/instinctlab/tasks/"
    "parkour/config/parkour_env_cfg.py::ROUGH_TERRAINS_CFG"
  ),
  "approximated_sub_terrains": (
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "square_gaps",
    "boxes",
    "mesh_boxes",
  ),
  "instinctlab_params": {
    "pyramid_stairs": {
      "step_height_range_m": (0.05, 0.23),
      "step_width_m": 0.3,
      "platform_width_m": 2.5,
    },
    "pyramid_stairs_inv": {
      "step_height_range_m": (0.05, 0.23),
      "step_width_m": 0.3,
      "platform_width_m": 2.5,
    },
    "square_gaps": {
      "gap_distance_range_m": (0.1, 0.4),
      "gap_depth_m": (0.4, 0.6),
      "platform_width_m": 2.5,
    },
    "boxes": {
      "num_obstacles": 20,
      "obstacle_width_range_m": (0.8, 1.5),
      "obstacle_height_range_m": (0.05, 0.45),
    },
    "mesh_boxes": {
      "box_height_mean_m": (0.1, 0.4),
      "box_height_range_m": 0.05,
      "box_length_mean_m": 0.4,
      "box_width_mean_m": 0.4,
    },
  },
}


def _apply_parkour_observation_contract(cfg: ManagerBasedRlEnvCfg) -> None:
  policy_asset = SceneEntityCfg(
    "robot",
    joint_names=TRAINING_JOINT_NAMES,
    preserve_order=True,
  )
  allowed_actor_terms = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
  )

  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.terms = {
      name: deepcopy(term)
      for name, term in group.terms.items()
      if name in allowed_actor_terms
    }
    group.enable_corruption = False
    group.history_length = 1
    for term_name, term in group.terms.items():
      term.params = dict(term.params or {})
      sensor_name = term.params.get("sensor_name")
      if sensor_name in PARKOUR_SCENE_SENSOR_REMAP:
        term.params["sensor_name"] = PARKOUR_SCENE_SENSOR_REMAP[sensor_name]
      if term_name in {"joint_pos", "joint_vel"}:
        term.params["asset_cfg"] = deepcopy(policy_asset)

  assert_no_stale_sensor_references(cfg)


def _apply_parkour_action_contract(cfg: ManagerBasedRlEnvCfg) -> None:
  contract = load_deploy_contract()
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=contract.action_scale_by_joint,
      use_default_offset=True,
    )
  }


def _apply_fixed_command_profile(
  cfg: ManagerBasedRlEnvCfg,
  *,
  command_x: float = DEFAULT_COMMAND_X,
) -> None:
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.heading_command = False
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.resampling_time_range = (1.0e9, 1.0e9)
  twist_cmd.ranges.lin_vel_x = (command_x, command_x)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None


def _remove_training_only_managers(cfg: ManagerBasedRlEnvCfg) -> None:
  # ``play_parkour.py`` owns ONNX observations/rewards/diagnostics directly.
  # Keep only deterministic reset events and minimal safety terminations so env.step
  # remains useful without inherited velocity rewards/sensors masking root causes.
  cfg.events = {
    name: term
    for name, term in cfg.events.items()
    if name in {"reset_base", "reset_robot_joints"}
  }
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
  }
  reset_base.params["velocity_range"] = {}

  reset_joints = cfg.events["reset_robot_joints"]
  reset_joints.params["position_range"] = (0.0, 0.0)
  reset_joints.params["velocity_range"] = (0.0, 0.0)
  reset_joints.params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=TRAINING_JOINT_NAMES,
    preserve_order=True,
  )

  cfg.rewards = {}
  cfg.metrics = {}
  cfg.curriculum = {}


def unitree_g1_parkour_flat_debug_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a deterministic flat-debug env for the torso-root parkour G1."""
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_parkour_robot_cfg()}
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 30.0 if not play else int(1e9)
  cfg.sim.njmax = 1500
  cfg.sim.nconmax = 256
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.viewer.body_name = "torso_link"

  _apply_parkour_observation_contract(cfg)
  _apply_parkour_action_contract(cfg)
  _apply_fixed_command_profile(cfg)
  _remove_training_only_managers(cfg)
  assert_no_stale_sensor_references(cfg)
  cfg.g1_parkour_flat_debug = True  # type: ignore[attr-defined]
  cfg.g1_parkour_policy_joint_names = TRAINING_JOINT_NAMES  # type: ignore[attr-defined]
  return cfg


def unitree_g1_parkour_obstacle_debug_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a deterministic low-block + shallow-gap parkour debug env."""
  cfg = unitree_g1_parkour_flat_debug_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_parkour_obstacle_robot_cfg()}
  cfg.g1_parkour_flat_debug = False  # type: ignore[attr-defined]
  cfg.g1_parkour_obstacle_debug = True  # type: ignore[attr-defined]
  cfg.g1_parkour_obstacle_geoms = PARKOUR_OBSTACLE_DEBUG_GEOMS  # type: ignore[attr-defined]
  cfg.g1_parkour_obstacle_contract = {  # type: ignore[attr-defined]
    "low_block_height_m": 0.05,
    "gap_width_m": 0.10,
    "target_distance_m": 3.0,
  }
  return cfg


def unitree_g1_parkour_complex_terrain_debug_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an InstinctLab-inspired complex terrain debug env.

  This task intentionally uses deterministic MJCF boxes instead of the
  procedural IsaacLab height-field generator.  It gives the depth renderer and
  policy loop a repeatable course with up/down stairs, a square-gap surrogate,
  discrete boxes, and mesh-box stepping stones.
  """
  cfg = unitree_g1_parkour_flat_debug_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_g1_parkour_complex_terrain_robot_cfg()}
  cfg.sim.nconmax = max(cfg.sim.nconmax, 512)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 512)
  cfg.g1_parkour_flat_debug = False  # type: ignore[attr-defined]
  cfg.g1_parkour_obstacle_debug = False  # type: ignore[attr-defined]
  cfg.g1_parkour_complex_terrain = True  # type: ignore[attr-defined]
  cfg.g1_parkour_complex_terrain_debug = True  # type: ignore[attr-defined]
  cfg.g1_parkour_complex_terrain_geoms = (  # type: ignore[attr-defined]
    PARKOUR_COMPLEX_TERRAIN_DEBUG_GEOMS
  )
  cfg.g1_parkour_route_waypoints = (  # type: ignore[attr-defined]
    PARKOUR_COMPLEX_TERRAIN_ROUTE_WAYPOINTS
  )
  cfg.g1_parkour_complex_terrain_contract = {  # type: ignore[attr-defined]
    "target_distance_m": 25.2,
    "up_stairs": {"steps": 5, "step_run_m": 0.36, "max_height_m": 0.30},
    "down_stairs": {"steps": 5, "step_run_m": 0.36, "max_height_m": 0.30},
    "second_stairs": {"steps": 4, "step_run_m": 0.42, "max_height_m": 0.28},
    "gap": {
      "platform_height_m": 0.11,
      "lower_strip_width_m": 0.36,
      "second_lower_strip_width_m": 0.44,
      "keeps_global_floor": True,
    },
    "box_field": {
      "discrete_boxes": 6,
      "mesh_style_boxes": 6,
      "height_range_m": (0.08, 0.18),
    },
    "route_waypoints": PARKOUR_COMPLEX_TERRAIN_ROUTE_WAYPOINTS,
    "instinctlab_reference": PARKOUR_COMPLEX_TERRAIN_INSTINCTLAB_REFERENCE,
  }
  return cfg


def unitree_g1_parkour_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the default non-debug G1 parkour play env on complex terrain."""
  cfg = unitree_g1_parkour_complex_terrain_debug_env_cfg(play=play)
  cfg.g1_parkour_official = True  # type: ignore[attr-defined]
  cfg.g1_parkour_complex_terrain_debug = False  # type: ignore[attr-defined]
  return cfg
