from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import diagnose_antifall_getup_rollout as diag


def test_antifall_getup_summary_requires_walk_disturb_recover_resume() -> None:
  records = [
    {
      "type": "metadata",
      "num_envs": 10,
      "task_id": "Unitree-G1-AntiFall-GetUp",
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"moving_command_rate": 1.0, "tracking_rate": 0.9},
      "root": {"fallen_rate": 0.0},
      "metrics": {"disturbance_count": 0.0, "recovery_success_count": 0.0, "recovery_latency": 0.0, "controllable_locomotion": 0.9},
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"moving_command_rate": 1.0, "tracking_rate": 0.2},
      "root": {"fallen_rate": 0.7},
      "metrics": {"disturbance_count": 0.2, "recovery_success_count": 0.0, "recovery_latency": 0.0, "controllable_locomotion": 0.2},
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"moving_command_rate": 1.0, "tracking_rate": 0.85},
      "root": {"fallen_rate": 0.0},
      "metrics": {"disturbance_count": 0.0, "recovery_success_count": 0.2, "recovery_latency": 1.2, "controllable_locomotion": 0.85},
    },
  ]

  summary = diag.summarize_records(records, success_threshold=0.8)

  assert summary["schema_version"] == diag.SCHEMA_VERSION
  assert summary["walk_disturb_recover_resume_gate"] is True
  assert summary["disturbance_count_estimate"] == 2
  assert summary["recovery_success_count_estimate"] == 2
  assert summary["max_fallen_rate"] == 0.7
  assert summary["final_controllable_rate"] == 0.85


def test_antifall_getup_summary_gates_on_tracking_not_controllable_locomotion() -> None:
  records = [
    {
      "type": "metadata",
      "num_envs": 8,
      "task_id": "Unitree-G1-AntiFall-GetUp",
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"moving_command_rate": 1.0, "tracking_rate": 0.92},
      "root": {"fallen_rate": 0.0},
      "metrics": {
        "disturbance_count": 0.0,
        "recovery_success_count": 0.0,
        "recovery_latency": 0.0,
        "controllable_locomotion": 0.58,
      },
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"moving_command_rate": 1.0, "tracking_rate": 0.18},
      "root": {"fallen_rate": 0.65},
      "metrics": {
        "disturbance_count": 0.25,
        "recovery_success_count": 0.0,
        "recovery_latency": 0.0,
        "controllable_locomotion": 0.22,
      },
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"moving_command_rate": 1.0, "tracking_rate": 0.84},
      "root": {"fallen_rate": 0.0},
      "metrics": {
        "disturbance_count": 0.0,
        "recovery_success_count": 0.25,
        "recovery_latency": 1.1,
        "controllable_locomotion": 0.29,
      },
    },
  ]

  summary = diag.summarize_records(records, success_threshold=0.8)

  assert summary["walk_disturb_recover_resume_gate"] is True
  assert summary["pre_disturbance_tracking_rate"] == pytest.approx(0.92)
  assert summary["post_disturbance_tracking_rate"] == pytest.approx(0.84)
  assert summary["final_tracking_rate"] == pytest.approx(0.84)
  assert summary["post_disturbance_controllable_rate"] == pytest.approx(0.29)
  assert summary["final_controllable_rate"] == pytest.approx(0.29)


def test_antifall_getup_forced_fall_options_are_recorded_in_metadata() -> None:
  args = diag.build_parser().parse_args(
    [
      "--agent",
      "zero",
      "--force-fall-step",
      "25",
      "--force-fall-prob",
      "0.75",
    ]
  )

  metadata = diag.build_metadata_record(args, num_envs=4, clip_actions=2.0)

  assert metadata["forced_fall_step"] == 25
  assert metadata["forced_fall_prob"] == pytest.approx(0.75)


def test_antifall_getup_force_fall_reset_marks_near_failure_disturbance(monkeypatch) -> None:
  calls = []

  def fake_reset_root_state_mixed(env, env_ids, **kwargs):
    calls.append((env, env_ids, kwargs))

  monkeypatch.setattr(diag.mdp, "reset_root_state_mixed", fake_reset_root_state_mixed)
  env = SimpleNamespace(num_envs=3, device="cpu")

  diag.force_fall_reset(env, prob=0.5)

  assert len(calls) == 1
  _, env_ids, kwargs = calls[0]
  assert env_ids.tolist() == [0, 1, 2]
  assert kwargs["hard_reset_prob"] == pytest.approx(0.5)
  assert kwargs["hard_pose_range"]["roll"][0] <= -2.0
  assert kwargs["hard_pose_range"]["pitch"][1] >= 2.0
