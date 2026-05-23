from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path

from mjlab.managers.observation_manager import ObservationGroupCfg


def test_default_getup_stays_no_demo_no_amp() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg
  from src.tasks.velocity.mdp.getup import rewards

  cfg = unitree_g1_getup_env_cfg("ground")

  assert "amp" not in cfg.observations
  assert not getattr(cfg, "getup_amp_enabled", False)
  assert all(term.func is not rewards.getup_demo_pose_reward for term in cfg.rewards.values())
  assert all(not name.startswith("amp") for name in cfg.rewards)


def test_amp_env_is_ground_only_opt_in_observation_group() -> None:
  from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_amp_env_cfg

  cfg = unitree_g1_getup_amp_env_cfg(demo_data_dir="/tmp/demo")

  assert cfg.getup_terrain == "ground"
  assert cfg.getup_amp_enabled is True
  assert cfg.getup_amp_demo_data_dir == "/tmp/demo"
  assert isinstance(cfg.observations["amp"], ObservationGroupCfg)
  assert cfg.observations["amp"].enable_corruption is False
  assert "amp" not in cfg.observations["actor"].terms
  assert "amp" not in cfg.observations["critic"].terms


def test_amp_algorithm_cfg_serializes_extra_fields() -> None:
  from src.tasks.velocity.config.g1_getup.rl_cfg import unitree_g1_getup_amp_ppo_runner_cfg

  cfg = unitree_g1_getup_amp_ppo_runner_cfg(
    demo_data_dir="/tmp/g1_getup_amp_fixture",
    manifest_path="/tmp/g1_getup_amp_fixture/manifest.json",
    amp_reward_scale=0.5,
  )
  serialized = asdict(cfg)
  algorithm = serialized["algorithm"]

  assert algorithm["class_name"] == "src.tasks.velocity.rl.getup_amp:GetupAmpPPO"
  assert algorithm["demo_data_dir"] == "/tmp/g1_getup_amp_fixture"
  assert algorithm["manifest_path"] == "/tmp/g1_getup_amp_fixture/manifest.json"
  assert algorithm["amp_reward_scale"] == 0.5
  assert algorithm["amp_obs_group"] == "amp"
  assert algorithm["discriminator_hidden_dims"] == (256, 128)


def test_amp_algorithm_cfg_uses_conservative_warm_start_updates() -> None:
  from src.tasks.velocity.config.g1_getup.rl_cfg import unitree_g1_getup_amp_ppo_runner_cfg

  cfg = unitree_g1_getup_amp_ppo_runner_cfg()

  assert cfg.algorithm.learning_rate == 1.0e-5
  assert cfg.algorithm.desired_kl == 0.001
  assert cfg.algorithm.amp_reward_scale == 0.1


def test_amp_task_is_registered_without_changing_default() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks, load_rl_cfg

  tasks = set(list_tasks())
  assert "Unitree-G1-GetUp" in tasks
  assert "Unitree-G1-GetUp-AMP" in tasks
  assert load_rl_cfg("Unitree-G1-GetUp").algorithm.class_name == "PPO"
  assert load_rl_cfg("Unitree-G1-GetUp-AMP").algorithm.class_name.endswith("GetupAmpPPO")


def test_new_amp_tests_and_fixtures_are_trackable() -> None:
  repo = Path(__file__).resolve().parents[2]
  paths = [
    "tests/tasks/test_g1_getup_amp_contract.py",
    "tests/tasks/test_g1_getup_amp_algorithm.py",
    "tests/scripts/test_prepare_g1_getup_amp_data.py",
    "tests/scripts/test_play_g1_getup_amp_data.py",
    "tests/fixtures/g1_getup_amp/valid_getup_canonical.npz",
  ]
  for rel in paths:
    result = subprocess.run(["git", "check-ignore", "-q", rel], cwd=repo)
    assert result.returncode == 1, f"{rel} is unexpectedly ignored"
