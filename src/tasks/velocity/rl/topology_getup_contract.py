"""Deploy/export contract helpers for topology get-up tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.sensor import CameraSensorCfg

from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.utils.os import dump_yaml

_SGI_VERSION = "sgi_v1"
_SGI_ANCHORS = ("trunk", "left_hand", "right_hand", "left_foot", "right_foot")
_SUPPORT_DEPTH_NAME = "support_depth"
_DEPTH_GROUP = "camera"
_DEPTH_TERM = "support_depth"
_DEPTH_NORMALIZATION_FRAME = "torso_link"
_DEPTH_MISSING_DATA_POLICY = "zeros"
_DEPTH_CUTOFF_DISTANCE = 1.5
_DEPTH_TIMEOUT_MS = 500
_DEPTH_RETAIN_LAST_VALID_FRAME = True
_DEPTH_ORGANIZED_POINTCLOUD = True
_DEPTH_POINTCLOUD_FIELD_NAMES = {"x": "x", "y": "y", "z": "z"}

_DEPLOY_OBSERVATION_ALIASES = {
  "command": "velocity_commands",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


def _camera_cfg(env: ManagerBasedRlEnv, sensor_name: str = _SUPPORT_DEPTH_NAME) -> CameraSensorCfg:
  for sensor in env.cfg.scene.sensors or ():
    if isinstance(sensor, CameraSensorCfg) and sensor.name == sensor_name:
      return sensor
  raise KeyError(f"Camera sensor '{sensor_name}' not found in env config")


def _actor_deploy_term_name(term_name: str) -> str:
  return _DEPLOY_OBSERVATION_ALIASES.get(term_name, term_name)


def get_support_geometry_metadata(env: ManagerBasedRlEnv) -> dict[str, Any]:
  cam_cfg = _camera_cfg(env)
  actor_terms = tuple(env.observation_manager.active_terms["actor"])
  camera_terms = tuple(env.observation_manager.active_terms[_DEPTH_GROUP])
  camera_term_cfg = env.cfg.observations[_DEPTH_GROUP].terms[_DEPTH_TERM]
  return {
    "support_geometry_interface_version": _SGI_VERSION,
    "support_geometry_anchor_names": list(_SGI_ANCHORS),
    "support_geometry_patch_shape": [cam_cfg.height, cam_cfg.width],
    "support_geometry_history_length": camera_term_cfg.history_length,
    "support_geometry_normalization_frame": _DEPTH_NORMALIZATION_FRAME,
    "support_geometry_missing_data_policy": _DEPTH_MISSING_DATA_POLICY,
    "support_geometry_depth_camera_contract": {
      "sensor_name": cam_cfg.name,
      "parent_body": cam_cfg.parent_body,
      "width": cam_cfg.width,
      "height": cam_cfg.height,
      "data_types": list(cam_cfg.data_types),
      "cutoff_distance": _DEPTH_CUTOFF_DISTANCE,
      "topic_name": "",
      "pointcloud_mode": "euclidean_norm",
      "timeout_ms": _DEPTH_TIMEOUT_MS,
      "retain_last_valid_frame": _DEPTH_RETAIN_LAST_VALID_FRAME,
      "organized_pointcloud": _DEPTH_ORGANIZED_POINTCLOUD,
      "pointcloud_field_names": dict(_DEPTH_POINTCLOUD_FIELD_NAMES),
    },
    "support_geometry_student_obs_groups": {
      "actor": [_actor_deploy_term_name(term) for term in actor_terms],
      _DEPTH_GROUP: list(camera_terms),
    },
  }


def _joint_ids_map(env: ManagerBasedRlEnv) -> list[int]:
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  return [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names
    if jname in joint_name_to_ctrl_id
  ]


def build_topology_getup_deploy_cfg(env: ManagerBasedRlEnv) -> dict[str, Any]:
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  base_metadata = get_base_metadata(env, "local")
  metadata = get_support_geometry_metadata(env)
  twist_cmd = env.cfg.commands["twist"]
  actor_terms = tuple(env.observation_manager.active_terms["actor"])
  camera_term_cfg = env.cfg.observations[_DEPTH_GROUP].terms[_DEPTH_TERM]
  camera_group = {
    _DEPTH_TERM: {
      "params": {
        "sensor_name": _SUPPORT_DEPTH_NAME,
        "expected_size": metadata["support_geometry_patch_shape"][0]
        * metadata["support_geometry_patch_shape"][1],
      },
      "clip": None,
      "scale": [1.0]
      * (metadata["support_geometry_patch_shape"][0] * metadata["support_geometry_patch_shape"][1]),
      "history_length": camera_term_cfg.history_length,
    }
  }
  deploy_cfg: dict[str, Any] = {
    "joint_ids_map": _joint_ids_map(env),
    "step_dt": env.step_dt,
    "stiffness": base_metadata["joint_stiffness"],
    "damping": base_metadata["joint_damping"],
    "default_joint_pos": base_metadata["default_joint_pos"],
    "commands": {
      "base_velocity": {
        "ranges": {
          "lin_vel_x": list(twist_cmd.ranges.lin_vel_x),
          "lin_vel_y": list(twist_cmd.ranges.lin_vel_y),
          "ang_vel_z": list(twist_cmd.ranges.ang_vel_z),
          "heading": None,
        }
      }
    },
    "actions": {
      "JointPositionAction": {
        "clip": None,
        "joint_names": [".*"],
        "scale": joint_action._scale[0].cpu().tolist()
        if hasattr(joint_action._scale, "cpu")
        else joint_action._scale,
        "offset": base_metadata["default_joint_pos"],
        "joint_ids": None,
      }
    },
    "observations": {
      "actor": {
        _actor_deploy_term_name(term): {
          "params": ({"command_name": "base_velocity"} if term == "command" else {}),
          "clip": None,
          "scale": [1.0] * int(env.observation_manager._group_obs_term_dim["actor"][idx][0]),
          "history_length": env.cfg.observations["actor"].history_length,
        }
        for idx, term in enumerate(actor_terms)
      },
      _DEPTH_GROUP: camera_group,
    },
    "support_geometry_interface": {
      "version": metadata["support_geometry_interface_version"],
      "anchor_names": metadata["support_geometry_anchor_names"],
      "patch_shape": metadata["support_geometry_patch_shape"],
      "history_length": metadata["support_geometry_history_length"],
      "normalization_frame": metadata["support_geometry_normalization_frame"],
      "missing_data_policy": metadata["support_geometry_missing_data_policy"],
      "depth_camera": metadata["support_geometry_depth_camera_contract"],
    },
  }
  return deploy_cfg


def write_topology_getup_deploy_yaml(env: ManagerBasedRlEnv, filename: str | Path) -> None:
  dump_yaml(Path(filename), build_topology_getup_deploy_cfg(env), sort_keys=False)
