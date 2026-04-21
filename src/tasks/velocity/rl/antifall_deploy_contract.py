"""Deploy/export contract helpers for G1 anti-fall tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.utils.os import dump_yaml

_DEPLOY_OBSERVATION_ALIASES = {
  "command": "velocity_commands",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


def _joint_ids_map(env: ManagerBasedRlEnv) -> list[int]:
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  # The deploy runtime reads lowstate / writes lowcmd by low-level motor index.
  # For the existing working G1 deploy target, that order is the natural actuated
  # joint order expected by the exported actor/action contract, not MuJoCo
  # actuator IDs. Using actuator.id here scrambles both observations and actions.
  return list(range(joint_action.action_dim))


def _actor_deploy_term_name(term_name: str) -> str:
  return _DEPLOY_OBSERVATION_ALIASES.get(term_name, term_name)


def _actor_term_dim(term_name: str, *, joint_dim: int) -> int:
  if term_name in ("base_ang_vel", "projected_gravity", "command"):
    return 3
  if term_name in ("joint_pos", "joint_vel", "actions"):
    return joint_dim
  raise KeyError(f"Unsupported anti-fall actor term: {term_name}")


def build_antifall_deploy_cfg(env: ManagerBasedRlEnv) -> dict[str, Any]:
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  base_metadata = get_base_metadata(env, "local")
  joint_dim = len(base_metadata["joint_names"])
  twist_cmd = env.cfg.commands["twist"]
  actor_terms = tuple(env.observation_manager.active_terms["actor"])
  actor_history = int(env.cfg.observations["actor"].history_length or 1)

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
      _actor_deploy_term_name(term): {
        "params": ({"command_name": "base_velocity"} if term == "command" else {}),
        "clip": None,
        "scale": [1.0] * _actor_term_dim(term, joint_dim=joint_dim),
        "history_length": actor_history,
      }
      for term in actor_terms
    },
  }
  return deploy_cfg


def write_antifall_deploy_yaml(env: ManagerBasedRlEnv, filename: str | Path) -> None:
  dump_yaml(Path(filename), build_antifall_deploy_cfg(env), sort_keys=False)
