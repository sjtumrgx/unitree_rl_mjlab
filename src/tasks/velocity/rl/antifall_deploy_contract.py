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
  if term_name == "getup_progress":
    return 5
  if term_name == "recovery_phase":
    return 1
  if term_name == "bfm_local_body_state":
    # Current AntiFall-GetUp 29-DoF G1 body-state layout uses the 30-body XML:
    # root height (1), root-relative positions without root ((30-1)*3), 6D
    # rotations (30*6), local linear velocities (30*3), and local angular
    # velocities (30*3).  Keep this synchronized with scripts.train's
    # _G1_ANTIFALL_29DOF_BODY_NAMES projection layout.
    body_dim = joint_dim + 1
    return 1 + (body_dim - 1) * 3 + body_dim * 6 + body_dim * 3 + body_dim * 3
  if term_name == "height_scan":
    return 187
  raise KeyError(f"Unsupported anti-fall actor term: {term_name}")


def _actor_term_history_length(env: ManagerBasedRlEnv, term_name: str) -> int:
  actor_group = env.cfg.observations["actor"]
  group_history = actor_group.history_length
  if group_history is not None:
    return int(group_history)

  term = actor_group.terms[term_name]
  term_history = getattr(term, "history_length", None)
  if term_history is None:
    return 1
  term_history = int(term_history)
  return term_history if term_history > 0 else 1


def build_antifall_deploy_cfg(env: ManagerBasedRlEnv) -> dict[str, Any]:
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)
  base_metadata = get_base_metadata(env, "local")
  joint_dim = len(base_metadata["joint_names"])
  twist_cmd = env.cfg.commands["twist"]
  actor_terms = tuple(env.observation_manager.active_terms["actor"])

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
        "history_length": _actor_term_history_length(env, term),
      }
      for term in actor_terms
    },
  }
  return deploy_cfg


def write_antifall_deploy_yaml(env: ManagerBasedRlEnv, filename: str | Path) -> None:
  dump_yaml(Path(filename), build_antifall_deploy_cfg(env), sort_keys=False)
