from __future__ import annotations

import re
from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg


_DEPLOY_OBSERVATION_ALIASES = {
  "command": "velocity_commands",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


def _deploy_observation_registry() -> set[str]:
  header = (
    Path(__file__).resolve().parents[2]
    / "deploy/include/isaaclab/envs/mdp/observations/observations.h"
  )
  text = header.read_text()
  return set(re.findall(r"REGISTER_OBSERVATION\(([^)]+)\)", text))


def _expected_deploy_names(actor_terms: tuple[str, ...]) -> tuple[str, ...]:
  return tuple(_DEPLOY_OBSERVATION_ALIASES.get(term, term) for term in actor_terms)


def test_antifall_actor_terms_map_to_registered_deploy_observations() -> None:
  deploy_registry = _deploy_observation_registry()

  for task_id in (
    "Unitree-G1-AntiFall-Stage0",
    "Unitree-G1-AntiFall-Stage1",
    "Unitree-G1-AntiFall-Stage2",
    "Unitree-G1-AntiFall-Stage3",
    "Unitree-G1-AntiFall-Stage4a",
    "Unitree-G1-AntiFall-Stage4b",
    "Unitree-G1-AntiFall-Benchmark",
  ):
    cfg = load_env_cfg(task_id)
    actor_terms = tuple(cfg.observations["actor"].terms)
    assert actor_terms == (
      "base_ang_vel",
      "projected_gravity",
      "command",
      "joint_pos",
      "joint_vel",
      "actions",
    )
    assert cfg.observations["actor"].history_length == 3
    deploy_names = _expected_deploy_names(actor_terms)
    assert set(deploy_names).issubset(deploy_registry)


def test_antifall_base_export_metadata_matches_actor_contract() -> None:
  cfg = load_env_cfg("Unitree-G1-AntiFall-Stage0")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    metadata = get_base_metadata(env, "local")
  finally:
    env.close()

  observation_names = tuple(metadata["observation_names"])
  assert observation_names == (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
  )
  assert _expected_deploy_names(observation_names) == (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
  )
