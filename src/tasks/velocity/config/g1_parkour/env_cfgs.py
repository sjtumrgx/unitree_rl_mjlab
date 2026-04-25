"""Flat MuJoCo debug task for the exported G1 parkour ONNX policy."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import get_g1_parkour_obstacle_robot_cfg, get_g1_parkour_robot_cfg
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from src.parkour.contract import (
  PARKOUR_SCENE_SENSOR_REMAP,
  TRAINING_JOINT_NAMES,
  assert_no_stale_sensor_references,
  load_deploy_contract,
)

PARKOUR_FLAT_DEBUG_TASK_ID = "Unitree-G1-Parkour-FlatDebug"
PARKOUR_OBSTACLE_DEBUG_TASK_ID = "Unitree-G1-Parkour-ObstacleDebug"
DEFAULT_COMMAND_X = 0.25
PARKOUR_OBSTACLE_DEBUG_GEOMS = (
  "robot/parkour_debug_low_block",
  "robot/parkour_debug_gap_near_lip",
  "robot/parkour_debug_gap_far_lip",
)


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
