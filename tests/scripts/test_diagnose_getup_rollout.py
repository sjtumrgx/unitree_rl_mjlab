from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import diagnose_getup_rollout as rollout


class _FakeAsset:
  body_names = ("torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
  joint_names = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
  )

  def __init__(self) -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    self.data = SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 1.1]]),
      root_link_quat_w=identity.reshape(1, 4),
      root_link_lin_vel_w=torch.tensor([[0.1, 0.2, 2.5]]),
      root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 3.0]]),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 1.2], [0.1, 0.1, 0.0], [0.1, -0.1, 0.0]]]),
      body_link_quat_w=identity.reshape(1, 1, 4).repeat(1, 3, 1),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -0.9]]),
      joint_pos=torch.zeros(1, 12),
      joint_vel=torch.zeros(1, 12),
      joint_acc=torch.zeros(1, 12),
      body_link_lin_vel_w=torch.zeros(1, 3, 3),
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
      "foot_geom_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.ones(1, 14, 1))),
      "hand_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 0.0]]))),
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
    self._host_getup_joint_position_target = torch.zeros(1, 12)
    self._host_getup_joint_position_target[:, :2] = torch.tensor([[1.1, -0.4]])
    self._host_getup_joint_target_ids = torch.tensor([0, 1])
    self._host_getup_curriculum_state = {
      "force_n": torch.tensor([80.0, 100.0]),
      "action_rescale": torch.tensor([0.98, 1.0]),
      "max_torso_height": torch.tensor([1.2, 0.3]),
      "episode_success": torch.tensor([True, False]),
      "episode_force_scale": torch.tensor([0.0, 1.0]),
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
  assert record["root"]["root_z"] == pytest.approx(1.1)
  assert record["root"]["root_z_min"] == pytest.approx(1.1)
  assert record["root"]["root_z_mean"] == pytest.approx(1.1)
  assert record["root"]["root_z_max"] == pytest.approx(1.1)
  assert record["root"]["torso_height"] == pytest.approx(1.2)
  assert record["root"]["torso_height_min"] == pytest.approx(1.2)
  assert record["root"]["torso_height_mean"] == pytest.approx(1.2)
  assert record["root"]["torso_height_max"] == pytest.approx(1.2)
  assert record["root"]["upright_alignment"] == pytest.approx(0.9)
  assert record["root"]["upright_alignment_min"] == pytest.approx(0.9)
  assert record["root"]["upright_alignment_mean"] == pytest.approx(0.9)
  assert record["root"]["upright_alignment_max"] == pytest.approx(0.9)
  assert record["curriculum_assist"]["getup_assist_force_n"] == 100.0
  assert record["curriculum_assist"]["getup_assist_force_n_min"] == 80.0
  assert record["curriculum_assist"]["getup_assist_force_n_mean"] == pytest.approx(90.0)
  assert record["curriculum_assist"]["getup_action_rescale"] == 1.0
  assert record["curriculum_assist"]["getup_action_rescale_min"] == pytest.approx(0.98)
  assert record["curriculum_assist"]["episode_success_latched_rate"] == pytest.approx(0.5)
  assert record["curriculum_assist"]["episode_force_scale_min"] == 0.0
  assert record["curriculum_assist"]["episode_force_scale_mean"] == pytest.approx(0.5)
  assert record["curriculum_assist"]["episode_force_scale_max"] == 1.0
  assert record["support"]["feet_contact_count"] == 1.0
  assert record["support"]["hand_contact_count"] == 1.0
  assert record["posture"]["foot_flatness_min"] == pytest.approx(1.0)
  assert record["posture"]["foot_heading_alignment_min"] == pytest.approx(1.0)
  assert record["posture"]["foot_geom_contact_spread_min"] == pytest.approx(1.0)
  assert record["posture"]["natural_leg_pose_error_mean"] is not None
  assert record["reward"]["terms"]["host_lift_progress"] == pytest.approx(0.4)
  assert record["reward"]["terms"]["host_task_reward"] == pytest.approx(0.7)
  assert record["termination"]["terms"]["unstable_state"] is True
  assert record["amp"]["obs_dim"] == 51.0


def test_rollout_step_record_computes_mid_getup_hand_push_fields() -> None:
  env = _FakeEnv()
  asset = env.scene["robot"]
  asset.data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.32], [0.1, 0.1, 0.0], [0.1, -0.1, 0.0]]])
  asset.data.body_link_lin_vel_w = torch.tensor([[[0.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
  asset.data.projected_gravity_b = torch.tensor([[0.3, 0.0, -0.35]])
  env.scene["hand_ground_contact"].data.found = torch.tensor([[[1.0], [0.0]]])

  record = rollout.build_step_record(
    env,
    task_id="Unitree-G1-GetUp",
    step_index=3,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )

  assert record["posture"]["mid_getup_env_count"] == 1.0
  assert record["posture"]["mid_getup_hand_contact_rate"] == pytest.approx(1.0)
  assert record["posture"]["mid_getup_hand_push_rate"] == pytest.approx(1.0)
  assert record["posture"]["mid_getup_torso_upward_velocity_mean"] == pytest.approx(0.25)


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


def test_rollout_step_record_uses_batch_maxima_and_batch_supportless_spike() -> None:
  env = _FakeEnv()
  asset = env.scene["robot"]
  asset.data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 1.4]])
  asset.data.root_link_lin_vel_w = torch.zeros(2, 3)
  asset.data.root_link_ang_vel_w = torch.zeros(2, 3)
  asset.data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.2]], [[0.0, 0.0, 1.1]]])
  asset.data.projected_gravity_b = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.8]])
  asset.data.joint_pos = torch.zeros(2, 2)
  env.scene["env_origins"] = torch.zeros(2, 3)
  env.scene["feet_ground_contact"].data.found = torch.tensor([[[1.0]], [[0.0]]])
  env.scene["support_body_contact"].data.found = torch.tensor([[[0.0]], [[0.0]]])
  env._host_getup_joint_position_target = torch.zeros(2, 2)

  record = rollout.build_step_record(
    env,
    task_id="Unitree-G1-GetUp",
    step_index=0,
    mode="play-like",
    raw_action=torch.zeros(2, 2),
    clipped_action=torch.zeros(2, 2),
    previous_clipped_action=None,
    rewards=torch.zeros(2),
    dones=torch.zeros(2),
    extras={},
    clip_actions=5.0,
  )
  summary = rollout.summarize_records([record])

  assert record["root"]["torso_height"] == pytest.approx(1.1)
  assert record["root"]["torso_height_min"] == pytest.approx(0.2)
  assert record["root"]["torso_height_mean"] == pytest.approx(0.65)
  assert record["root"]["upright_alignment"] == pytest.approx(0.8)
  assert record["root"]["upright_alignment_min"] == pytest.approx(-0.5)
  assert record["root"]["supportless_height_spike"] is True
  assert summary["max_torso_height"] == pytest.approx(1.1)
  assert summary["risk_flags"]["supportless_height_spike"] is True


def test_rollout_summary_reports_single_episode_success_rate() -> None:
  metadata = {
    "type": "metadata",
    "num_envs": 4,
  }
  first = rollout.build_step_record(
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
  second = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=1,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  first["metrics"]["getup_success_count"] = 0.25
  first["metrics"]["getup_upright"] = 0.25
  second["metrics"]["getup_success_count"] = 0.5
  second["metrics"]["getup_upright"] = 0.5

  summary = rollout.summarize_records([metadata, first, second])

  assert summary["success"]["success_events_per_env"] == pytest.approx(0.75)
  assert summary["success"]["single_episode_success_rate"] == pytest.approx(0.75)
  assert summary["success"]["success_count_estimate"] == 3
  assert summary["success"]["num_envs"] == 4
  assert summary["success"]["max_getup_upright_rate"] == pytest.approx(0.5)
  assert summary["success"]["final_getup_upright_rate"] == pytest.approx(0.5)
  assert "posture" in summary
  assert "min_foot_flatness" in summary["posture"]
  assert "max_hand_contact_count" in summary["posture"]


def test_rollout_summary_reports_final_and_standing_phase_posture() -> None:
  metadata = {"type": "metadata", "num_envs": 4}
  early = rollout.build_step_record(
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
  final = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=1,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  early["root"]["torso_height_mean"] = 0.2
  early["root"]["upright_alignment_mean"] = 0.1
  early["posture"]["foot_flatness_min"] = 0.0
  early["posture"]["foot_heading_alignment_min"] = 0.0
  early["posture"]["foot_geom_contact_spread_min"] = 0.0
  early["posture"]["natural_leg_pose_error_max"] = 2.0
  early["support"]["hand_contact_count"] = 2.0
  final["root"]["torso_height_mean"] = 0.62
  final["root"]["upright_alignment_mean"] = 0.95
  final["posture"]["foot_flatness_min"] = 0.8
  final["posture"]["foot_heading_alignment_min"] = 0.7
  final["posture"]["foot_geom_contact_spread_min"] = 0.75
  final["posture"]["natural_leg_pose_error_max"] = 0.25
  final["support"]["hand_contact_count"] = 0.0

  summary = rollout.summarize_records([metadata, early, final])

  assert summary["posture"]["min_foot_flatness"] == 0.0
  assert summary["posture"]["standing_min_foot_flatness"] == pytest.approx(0.8)
  assert summary["posture"]["final_foot_flatness_min"] == pytest.approx(0.8)
  assert summary["posture"]["standing_min_foot_heading_alignment"] == pytest.approx(0.7)
  assert summary["posture"]["standing_min_foot_geom_contact_spread"] == pytest.approx(0.75)
  assert summary["posture"]["standing_max_natural_leg_pose_error"] == pytest.approx(0.25)
  assert summary["posture"]["standing_records"] == 1
  assert summary["posture"]["final_hand_contact_count"] == 0.0
  assert summary["risk_flags"]["standing_foot_flatness_lt_0_6"] is False
  assert summary["risk_flags"]["standing_foot_heading_alignment_lt_0_6"] is False
  assert summary["risk_flags"]["standing_foot_geom_contact_spread_lt_0_5"] is False


def test_rollout_summary_uses_last_standing_frame_before_timeout_reset() -> None:
  metadata = {"type": "metadata", "num_envs": 4}
  standing = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=598,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  reset = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=599,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([1]),
    extras={},
    clip_actions=5.0,
  )
  standing["root"]["torso_height_mean"] = 0.8
  standing["root"]["upright_alignment_mean"] = 0.99
  standing["posture"]["standing_env_count"] = 4.0
  standing["posture"]["standing_foot_flatness_min"] = 0.7
  standing["posture"]["standing_foot_heading_alignment_min"] = 0.8
  standing["posture"]["standing_foot_geom_contact_spread_min"] = 0.75
  standing["posture"]["standing_natural_leg_pose_error_max"] = 0.2
  standing["posture"]["standing_hand_contact_rate"] = 0.0
  reset["root"]["torso_height_mean"] = 0.2
  reset["root"]["upright_alignment_mean"] = 0.0
  reset["support"]["hand_contact_count"] = 2.0
  reset["posture"]["standing_env_count"] = 0.0

  summary = rollout.summarize_records([metadata, standing, reset])

  assert summary["posture"]["final_hand_contact_count"] == 2.0
  assert summary["posture"]["last_standing_hand_contact_rate"] == 0.0
  assert summary["posture"]["last_standing_foot_flatness_min"] == pytest.approx(0.7)
  assert summary["risk_flags"]["final_hand_contact_gt_0"] is True
  assert summary["risk_flags"]["last_standing_hand_contact_gt_0"] is False


def test_rollout_summary_reports_mid_getup_hand_push_phase() -> None:
  metadata = {"type": "metadata", "num_envs": 4}
  mid_push = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=10,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  standing = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=100,
    mode="play-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  mid_push["root"]["torso_height_mean"] = 0.32
  mid_push["root"]["upright_alignment_mean"] = 0.35
  mid_push["support"]["hand_contact_count"] = 2.0
  mid_push["posture"]["mid_getup_env_count"] = 4.0
  mid_push["posture"]["mid_getup_hand_contact_rate"] = 0.75
  mid_push["posture"]["mid_getup_hand_push_rate"] = 0.5
  mid_push["posture"]["mid_getup_torso_upward_velocity_mean"] = 0.2
  standing["root"]["torso_height_mean"] = 0.7
  standing["root"]["upright_alignment_mean"] = 0.95
  standing["support"]["hand_contact_count"] = 0.0
  standing["posture"]["mid_getup_env_count"] = 0.0

  summary = rollout.summarize_records([metadata, mid_push, standing])

  assert summary["posture"]["mid_getup_record_count"] == 1
  assert summary["posture"]["mid_getup_env_max_count"] == 4.0
  assert summary["posture"]["mean_mid_getup_hand_contact_rate"] == pytest.approx(0.75)
  assert summary["posture"]["mean_mid_getup_hand_push_rate"] == pytest.approx(0.5)
  assert summary["posture"]["max_mid_getup_hand_push_rate"] == pytest.approx(0.5)
  assert summary["posture"]["mean_mid_getup_torso_upward_velocity"] == pytest.approx(0.2)
  assert summary["risk_flags"]["mean_mid_getup_hand_contact_rate_lt_0_2"] is False
  assert summary["risk_flags"]["mean_mid_getup_hand_push_rate_lt_0_2"] is False


def test_rollout_summary_splits_train_success_by_assist_episode_scale() -> None:
  metadata = {
    "type": "metadata",
    "num_envs": 4,
  }
  assisted = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=0,
    mode="train-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  no_assist = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=1,
    mode="train-like",
    raw_action=torch.tensor([[0.0, 0.0]]),
    clipped_action=torch.tensor([[0.0, 0.0]]),
    previous_clipped_action=None,
    rewards=torch.tensor([0.0]),
    dones=torch.tensor([0]),
    extras={},
    clip_actions=5.0,
  )
  assisted["metrics"]["getup_success_count"] = 0.25
  assisted["metrics"]["getup_upright"] = 0.25
  assisted["curriculum_assist"]["episode_force_scale_mean"] = 1.0
  assisted["curriculum_assist"]["episode_force_scale_min"] = 1.0
  assisted["curriculum_assist"]["episode_force_scale_max"] = 1.0
  no_assist["metrics"]["getup_success_count"] = 0.5
  no_assist["metrics"]["getup_upright"] = 0.5
  no_assist["curriculum_assist"]["episode_force_scale_mean"] = 0.0
  no_assist["curriculum_assist"]["episode_force_scale_min"] = 0.0
  no_assist["curriculum_assist"]["episode_force_scale_max"] = 0.0

  summary = rollout.summarize_records([metadata, assisted, no_assist])

  assert summary["success"]["success_count_estimate"] == 3
  assert summary["success_by_assist"]["assisted"]["success_count_estimate"] == 1
  assert summary["success_by_assist"]["assisted"]["max_getup_upright_rate"] == pytest.approx(0.25)
  assert summary["success_by_assist"]["no_assist"]["success_count_estimate"] == 2
  assert summary["success_by_assist"]["no_assist"]["max_getup_upright_rate"] == pytest.approx(0.5)


def test_rollout_summary_splits_mixed_batch_success_by_per_env_assist_mask() -> None:
  metadata = {
    "type": "metadata",
    "num_envs": 4,
  }
  step = rollout.build_step_record(
    _FakeEnv(),
    task_id="Unitree-G1-GetUp",
    step_index=0,
    mode="train-like",
    raw_action=torch.zeros(4, 2),
    clipped_action=torch.zeros(4, 2),
    previous_clipped_action=None,
    rewards=torch.zeros(4),
    dones=torch.zeros(4),
    extras={},
    clip_actions=5.0,
  )
  step["curriculum_assist"]["episode_force_scale_mean"] = 0.5
  step["assist_success_split"] = {
    "assisted": {
      "env_count": 2,
      "success_events": 0.0,
      "upright_count": 0.0,
      "upright_rate": 0.0,
    },
    "no_assist": {
      "env_count": 2,
      "success_events": 1.0,
      "upright_count": 1.0,
      "upright_rate": 0.5,
    },
  }
  step["metrics"]["getup_success_count"] = 0.25
  step["metrics"]["getup_upright"] = 0.25

  summary = rollout.summarize_records([metadata, step])

  assert summary["success"]["success_count_estimate"] == 1
  assert summary["success_by_assist"]["assisted"]["success_count_estimate"] == 0
  assert summary["success_by_assist"]["assisted"]["single_episode_success_rate"] == 0.0
  assert summary["success_by_assist"]["no_assist"]["success_count_estimate"] == 1
  assert summary["success_by_assist"]["no_assist"]["single_episode_success_rate"] == pytest.approx(0.5)
  assert summary["success_by_assist"]["no_assist"]["max_getup_upright_rate"] == pytest.approx(0.5)



def test_rollout_summary_counts_play_like_as_no_assist_success() -> None:
  metadata = {"type": "metadata", "num_envs": 4}
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
  step["assist_success_split"] = None
  step["curriculum_assist"]["episode_force_scale_mean"] = None
  step["metrics"]["getup_success_count"] = 0.5
  step["metrics"]["getup_upright"] = 0.5

  summary = rollout.summarize_records([metadata, step])

  assert summary["success"]["success_count_estimate"] == 2
  assert summary["success_by_assist"]["assisted"]["success_count_estimate"] == 0
  assert summary["success_by_assist"]["assisted"]["env_count"] == 0
  assert summary["success_by_assist"]["no_assist"]["env_count"] == 4
  assert summary["success_by_assist"]["no_assist"]["success_count_estimate"] == 2
  assert summary["success_by_assist"]["no_assist"]["single_episode_success_rate"] == pytest.approx(0.5)

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


def test_make_trained_policy_uses_compatible_actor_input_expansion(monkeypatch, tmp_path: Path) -> None:
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  captured = {}

  class _FakeRunner:
    def __init__(self, env, agent_cfg, log_dir, device):
      captured["runner_init"] = {
        "env": env,
        "agent_cfg": agent_cfg,
        "log_dir": log_dir,
        "device": device,
      }

    def get_inference_policy(self, device):
      captured["inference_device"] = device
      return "policy"

  def _fake_load_actor(runner, resume_path, *, load_cfg, map_location):
    captured["load_actor"] = {
      "runner": runner,
      "resume_path": resume_path,
      "load_cfg": load_cfg,
      "map_location": map_location,
    }

  monkeypatch.setattr(rollout, "asdict", lambda cfg: {"logger": "wandb", "upload_model": True})
  monkeypatch.setattr("mjlab.tasks.registry.load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(
    "scripts.train._load_actor_with_compatible_input_expansion",
    _fake_load_actor,
  )
  args = SimpleNamespace(
    task_id="Unitree-G1-GetUp",
    checkpoint_file=str(checkpoint),
    device="cuda:0",
  )

  policy, runner = rollout._make_trained_policy(args, env="env", agent_cfg=SimpleNamespace())

  assert policy == "policy"
  assert isinstance(runner, _FakeRunner)
  assert captured["runner_init"]["agent_cfg"]["logger"] == "tensorboard"
  assert captured["runner_init"]["agent_cfg"]["upload_model"] is False
  assert captured["load_actor"]["resume_path"] == checkpoint
  assert captured["load_actor"]["load_cfg"] == {"actor": True}
  assert captured["load_actor"]["map_location"] == "cuda:0"
  assert captured["inference_device"] == "cuda:0"
