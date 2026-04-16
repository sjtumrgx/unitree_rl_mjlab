import json
import subprocess
import sys
from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from src.tasks.velocity import mdp

EXPECTED_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)
EXPECTED_CRITIC_HELPERS = (
  "disturbance_metadata",
  "recovery_features",
)
EXPECTED_REWARDS = (
  "upright_recoverability",
  "recovery_quality",
  "standing_stability",
  "recovery_completion_bonus",
)
EXPECTED_METRICS = (
  "disturbance_window_active",
  "disturbance_magnitude",
  "controllable_locomotion",
  "disturbance_count",
  "recovery_success_count",
  "recovery_latency",
)


def _env(task_id: str, *, play: bool = False):
  return load_env_cfg(task_id, play=play)


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def _assert_antifall_helpers_wired(cfg) -> None:
  for term_name in EXPECTED_CRITIC_HELPERS:
    assert term_name in cfg.observations["critic"].terms
  for term_name in EXPECTED_REWARDS:
    assert term_name in cfg.rewards
  for term_name in EXPECTED_METRICS:
    assert term_name in cfg.metrics
  assert cfg.events["reset_base"].func is mdp.reset_root_state_mixed


def test_antifall_actor_observation_contract_is_scaffolded() -> None:
  for task_id in (
    "Unitree-G1-AntiFall-Stage0",
    "Unitree-G1-AntiFall-Stage1",
    "Unitree-G1-AntiFall-Stage2",
    "Unitree-G1-AntiFall-Stage3",
    "Unitree-G1-AntiFall-Stage4a",
    "Unitree-G1-AntiFall-Stage4b",
    "Unitree-G1-AntiFall-Benchmark",
  ):
    cfg = _env(task_id)
    assert tuple(cfg.observations["actor"].terms) == EXPECTED_ACTOR_TERMS
    assert cfg.observations["actor"].history_length == 3


def test_stage0_stage1_stage2_contracts_progress_monotonically() -> None:
  stage0 = _env("Unitree-G1-AntiFall-Stage0")
  stage1 = _env("Unitree-G1-AntiFall-Stage1")
  stage2 = _env("Unitree-G1-AntiFall-Stage2")

  assert stage0.scene.terrain is not None
  assert stage1.scene.terrain is not None
  assert stage2.scene.terrain is not None
  assert stage0.scene.terrain.terrain_type == "plane"
  assert stage1.scene.terrain.terrain_type == "plane"
  assert stage2.scene.terrain.terrain_type == "plane"
  _assert_antifall_helpers_wired(stage0)
  _assert_antifall_helpers_wired(stage1)
  _assert_antifall_helpers_wired(stage2)

  assert "push_robot" not in stage0.events
  assert "push_robot" in stage1.events
  assert "push_robot" in stage2.events
  assert stage1.events["push_robot"].func is mdp.push_by_setting_velocity_with_history
  assert stage2.events["push_robot"].func is mdp.push_by_setting_velocity_with_history

  assert stage1.events["push_robot"].interval_range_s == (4.0, 6.0)
  assert stage2.events["push_robot"].interval_range_s == (3.0, 5.0)
  assert stage2.events["reset_base"].params["hard_reset_prob"] == 0.2
  assert stage2.events["reset_base"].params["hard_pose_range"]["roll"] == (-0.35, 0.35)
  assert stage2.events["reset_base"].params["hard_pose_range"]["pitch"] == (-0.35, 0.35)


def test_stage3_switches_to_rough_terrain_without_actor_height_scan() -> None:
  stage3 = _env("Unitree-G1-AntiFall-Stage3")

  assert stage3.scene.terrain is not None
  assert stage3.scene.terrain.terrain_type == "generator"
  _assert_antifall_helpers_wired(stage3)
  assert "push_robot" not in stage3.events
  assert "height_scan" not in stage3.observations["actor"].terms
  assert "height_scan" in stage3.observations["critic"].terms


def test_stage4a_and_stage4b_keep_specialized_hazard_scaffolds_isolated() -> None:
  stage4a = _env("Unitree-G1-AntiFall-Stage4a")
  stage4b = _env("Unitree-G1-AntiFall-Stage4b")

  _assert_antifall_helpers_wired(stage4a)
  _assert_antifall_helpers_wired(stage4b)
  assert stage4a.events["foot_friction"].params["ranges"] == (0.05, 0.35)

  assert stage4b.events["push_robot"].func is mdp.push_by_setting_velocity_with_history
  trip_profile = stage4b.events["push_robot"].params["velocity_range"]
  assert trip_profile["x"][0] > 0.0
  assert trip_profile["pitch"][0] > 0.0
  assert abs(trip_profile["y"][0]) <= 0.1
  assert abs(trip_profile["yaw"][1]) <= 0.2


def test_benchmark_cfg_disables_training_randomization() -> None:
  benchmark = _env("Unitree-G1-AntiFall-Benchmark")

  _assert_antifall_helpers_wired(benchmark)
  assert "push_robot" not in benchmark.events
  assert benchmark.curriculum == {}
  assert benchmark.observations["actor"].enable_corruption is False
  assert benchmark.events["foot_friction"].params["ranges"] == (1.0, 1.0)
  assert benchmark.events["encoder_bias"].params["bias_range"] == (0.0, 0.0)


def test_benchmark_cli_wrapper_emits_expected_contracts() -> None:
  repo_root = _repo_root()
  scenarios = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_antifall.py",
      "scenarios",
      "Unitree-G1-AntiFall-Benchmark",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(scenarios.stdout)
  assert payload["stage_name"] == "benchmark"
  assert len(payload["scenarios"]) == 12

  smoke = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_antifall.py",
      "smoke-command",
      "Unitree-G1-AntiFall-Stage4b",
      "--seed",
      "7",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  smoke_payload = json.loads(smoke.stdout)
  assert smoke_payload["command"][0] == sys.executable
  assert smoke_payload["command"][1] == "scripts/train.py"
  assert smoke_payload["command"][2] == "Unitree-G1-AntiFall-Stage4b"
