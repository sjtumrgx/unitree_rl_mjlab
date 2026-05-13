from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_getup_rollout as rollout


class _FakeAsset:
  body_names = ("torso_link",)

  def __init__(self) -> None:
    self.data = SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 1.1]]),
      root_link_lin_vel_w=torch.tensor([[0.1, 0.2, 2.5]]),
      root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 3.0]]),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 1.2]]]),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -0.9]]),
      joint_pos=torch.tensor([[0.2, -0.1]]),
      joint_vel=torch.zeros(1, 2),
      joint_acc=torch.zeros(1, 2),
    )


class _FakeEnv:
  def __init__(self) -> None:
    processed = SimpleNamespace(
      _processed_actions=torch.tensor([[0.5, -0.2]]),
      _raw_actions=torch.tensor([[1.0, -1.0]]),
    )
    self.scene = {
      "robot": _FakeAsset(),
      "env_origins": torch.zeros(1, 3),
      "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 0.0]]))),
      "support_body_contact": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0, 1.0]]))),
    }
    self.cfg = SimpleNamespace(events={"getup_assist_force": object()})
    self.action_manager = SimpleNamespace(_terms={"joint_pos": processed})
    self.reward_manager = SimpleNamespace(
      _term_names=["host_lift_progress", "host_task_reward", "untracked"],
      _step_reward=torch.tensor([[0.4, 0.7, 9.0]]),
    )
    self.termination_manager = SimpleNamespace(
      _term_dones={
        "time_out": torch.tensor([False]),
        "unstable_state": torch.tensor([True]),
      }
    )
    self.metrics_manager = SimpleNamespace(
      _term_names=["getup_upright"],
      _step_values=torch.tensor([[1.0]]),
    )
    self._host_getup_joint_position_target = torch.tensor([[1.3, -0.4]])
    self._host_getup_joint_target_ids = torch.tensor([0, 1])
    self._host_getup_curriculum_state = {
      "force_n": torch.tensor([100.0]),
      "action_rescale": torch.tensor([1.0]),
      "max_torso_height": torch.tensor([1.2]),
    }


def test_rollout_step_record_contains_required_debug_fields() -> None:
  record = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=3,
    mode="train-like",
    raw_action=torch.tensor([[2.0, -3.0]]),
    clipped_action=torch.tensor([[1.0, -1.0]]),
    previous_clipped_action=torch.zeros(1, 2),
    rewards=torch.tensor([1.1]),
    dones=torch.tensor([1]),
    extras={"time_outs": torch.tensor([False])},
    clip_actions=1.0,
    amp_stats={"obs_dim": 51.0, "reward_mean": 0.2, "policy_score": 0.4, "manifest_path": "m.json"},
  )

  assert record["schema_version"] == rollout.SCHEMA_VERSION
  assert record["action"]["raw_max_abs"] == 3.0
  assert record["action"]["clipped_max_abs"] == 1.0
  assert record["action"]["processed_max_abs"] == 0.5
  assert record["target"]["joint_target_delta_max"] == pytest.approx(1.1)
  assert record["root"]["root_vertical_velocity"] == 2.5
  assert record["support"]["feet_contact_count"] == 1.0
  assert record["reward"]["terms"]["host_lift_progress"] == pytest.approx(0.4)
  assert record["reward"]["terms"]["host_task_reward"] == pytest.approx(0.7)
  assert record["termination"]["terms"]["unstable_state"] is True
  assert record["amp"]["obs_dim"] == 51.0


def test_rollout_summary_flags_action_spike_and_ballistic_supportless_height() -> None:
  step = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=0,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  step["support"]["feet_contact_count"] = 0.0

  summary = rollout.summarize_records([step])

  assert summary["risk_flags"]["target_delta_gt_1rad"] is True
  assert summary["max_root_upward_velocity"] == 2.5
  assert summary["max_root_vertical_speed"] == 2.5
  assert summary["risk_flags"]["upward_velocity_gt_2mps"] is True
  assert summary["risk_flags"]["vertical_speed_gt_2mps"] is True
  # Compatibility field remains absolute speed for older report consumers.
  assert summary["risk_flags"]["vertical_velocity_gt_2mps"] is True
  assert summary["risk_flags"]["supportless_height_spike"] is True


def test_rollout_summary_distinguishes_downward_fall_from_upward_pop() -> None:
  step = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=0,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  step["target"]["joint_target_delta_max"] = 0.0
  step["root"]["root_vertical_velocity"] = -2.5
  step["root"]["torso_height"] = 0.2
  step["support"]["feet_contact_count"] = 1.0

  summary = rollout.summarize_records([step])

  assert summary["max_root_upward_velocity"] == 0.0
  assert summary["max_root_vertical_speed"] == 2.5
  assert summary["risk_flags"]["upward_velocity_gt_2mps"] is False
  assert summary["risk_flags"]["vertical_speed_gt_2mps"] is True


def test_rollout_main_writes_structured_blocker_json(monkeypatch, tmp_path: Path) -> None:
  def _raise(_args):
    raise RuntimeError("sim unavailable")

  monkeypatch.setattr(rollout, "_run_rollout_records", _raise)
  output = tmp_path / "blocked.json"

  rc = rollout.main(["Unitree-G1-GetUp", "--output", str(output), "--steps", "1"])

  assert rc == 2
  payload = json.loads(output.read_text())
  assert payload["schema_version"] == rollout.SCHEMA_VERSION
  assert payload["status"] == "blocked"
  assert payload["blocker"]["phase"] == "rollout"
  assert payload["blocker"]["exception_type"] == "RuntimeError"
  assert "sim unavailable" in payload["blocker"]["message"]
