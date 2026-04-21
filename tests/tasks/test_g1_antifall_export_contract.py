from __future__ import annotations

from dataclasses import asdict
import re
from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl.antifall_deploy_contract import build_antifall_deploy_cfg
from src.tasks.velocity.rl.antifall_runner import AntiFallOnPolicyRunner


_DEPLOY_OBSERVATION_ALIASES = {
  "command": "velocity_commands",
  "joint_pos": "joint_pos_rel",
  "joint_vel": "joint_vel_rel",
  "actions": "last_action",
}


def _velocity_joint_ids_map() -> list[int]:
  velocity_deploy_yaml = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "robots"
    / "g1"
    / "config"
    / "policy"
    / "velocity"
    / "v0"
    / "params"
    / "deploy.yaml"
  )
  import yaml

  return list(yaml.safe_load(velocity_deploy_yaml.read_text())["joint_ids_map"])


def _deploy_observation_registry() -> set[str]:
  header = (
    Path(__file__).resolve().parents[2]
    / "deploy/include/isaaclab/envs/mdp/observations/observations.h"
  )
  text = header.read_text()
  return set(re.findall(r"REGISTER_OBSERVATION\(([^)]+)\)", text))


def _expected_deploy_names(actor_terms: tuple[str, ...]) -> tuple[str, ...]:
  return tuple(_DEPLOY_OBSERVATION_ALIASES.get(term, term) for term in actor_terms)


def _build_runner(task_id: str, tmp_path: Path) -> AntiFallOnPolicyRunner:
  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 1
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=None)
  agent_cfg = asdict(load_rl_cfg(task_id))
  agent_cfg["logger"] = "tensorboard"
  agent_cfg["upload_model"] = False
  return AntiFallOnPolicyRunner(env, agent_cfg, str(tmp_path), "cpu")


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


def test_antifall_tasks_use_custom_runner() -> None:
  for task_id in (
    "Unitree-G1-AntiFall-Stage0",
    "Unitree-G1-AntiFall-Stage1",
    "Unitree-G1-AntiFall-Stage2",
    "Unitree-G1-AntiFall-Stage3",
    "Unitree-G1-AntiFall-Stage4a",
    "Unitree-G1-AntiFall-Stage4b",
    "Unitree-G1-AntiFall-Benchmark",
  ):
    assert load_runner_cls(task_id) is AntiFallOnPolicyRunner


def test_antifall_deploy_yaml_template_matches_actor_contract() -> None:
  cfg = load_env_cfg("Unitree-G1-AntiFall-Stage4b")
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    deploy_cfg = build_antifall_deploy_cfg(env)
  finally:
    env.close()

  assert tuple(deploy_cfg["observations"]) == (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
  )
  assert "gait_phase" not in deploy_cfg["observations"]
  assert {
    term_cfg["history_length"] for term_cfg in deploy_cfg["observations"].values()
  } == {3}
  assert deploy_cfg["observations"]["velocity_commands"]["params"] == {
    "command_name": "base_velocity"
  }
  assert deploy_cfg["joint_ids_map"] == _velocity_joint_ids_map()


def test_antifall_onpolicy_runner_save_exports_deploy_yaml(tmp_path: Path) -> None:
  runner = _build_runner("Unitree-G1-AntiFall-Stage4b", tmp_path)
  try:
    runner.logger.logger_type = "tensorboard"
    model_path = tmp_path / "model_0.pt"
    runner.save(str(model_path))
    assert model_path.exists()
    deploy_yaml = tmp_path / "params" / "deploy.yaml"
    assert deploy_yaml.exists()
    import yaml

    payload = yaml.safe_load(deploy_yaml.read_text())
    assert payload["joint_ids_map"] == _velocity_joint_ids_map()
    assert tuple(payload["observations"]) == (
      "base_ang_vel",
      "projected_gravity",
      "velocity_commands",
      "joint_pos_rel",
      "joint_vel_rel",
      "last_action",
    )
  finally:
    runner.env.close()
