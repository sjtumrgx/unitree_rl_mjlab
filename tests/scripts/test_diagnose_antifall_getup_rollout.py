from __future__ import annotations

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
