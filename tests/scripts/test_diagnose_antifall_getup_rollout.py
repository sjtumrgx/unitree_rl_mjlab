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


def test_antifall_getup_summary_reports_action_and_target_extrema() -> None:
  records = [
    {
      "type": "metadata",
      "num_envs": 4,
      "task_id": "Unitree-G1-AntiFall-GetUp",
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"tracking_rate": 0.9},
      "root": {"fallen_rate": 0.0},
      "metrics": {"disturbance_count": 0.0, "recovery_success_count": 0.0, "controllable_locomotion": 0.9},
      "action": {"raw_max_abs": 0.25, "clipped_max_abs": 0.25, "processed_max_abs": 0.1},
      "target": {"joint_target_delta_max": 0.3, "joint_target_abs_max": 0.6},
      "assist": {"force_z_max": 10.0, "active_rate": 0.25},
    },
    {
      "type": "step",
      "status": "ok",
      "command": {"tracking_rate": 0.1},
      "root": {"fallen_rate": 1.0},
      "metrics": {"disturbance_count": 0.25, "recovery_success_count": 0.0, "controllable_locomotion": 0.1},
      "action": {"raw_max_abs": 2.0, "clipped_max_abs": 1.0, "processed_max_abs": 0.8},
      "target": {"joint_target_delta_max": 1.2, "joint_target_abs_max": 1.4},
      "assist": {"force_z_max": 75.0, "active_rate": 0.5},
    },
  ]

  summary = diag.summarize_records(records)

  assert summary["max_action_raw_abs"] == pytest.approx(2.0)
  assert summary["max_action_clipped_abs"] == pytest.approx(1.0)
  assert summary["max_action_processed_abs"] == pytest.approx(0.8)
  assert summary["max_joint_target_delta"] == pytest.approx(1.2)
  assert summary["max_joint_target_abs"] == pytest.approx(1.4)
  assert summary["max_assist_force_z"] == pytest.approx(75.0)
  assert summary["max_assist_active_rate"] == pytest.approx(0.5)
  assert summary["risk_flags"]["assist_active_but_no_recovery_success"] is True


def test_antifall_getup_summary_uses_explicit_forced_fall_step_as_disturbance_boundary() -> None:
  records = [
    {
      "type": "metadata",
      "num_envs": 4,
      "task_id": "Unitree-G1-AntiFall-GetUp",
      "forced_fall_step": 3,
    },
    {
      "type": "step",
      "step": 0,
      "command": {"tracking_rate": 0.2},
      "root": {"fallen_rate": 0.25},
      "metrics": {"disturbance_count": 0.25, "recovery_success_count": 0.0, "controllable_locomotion": 0.2},
    },
    {
      "type": "step",
      "step": 1,
      "command": {"tracking_rate": 0.9},
      "root": {"fallen_rate": 0.0},
      "metrics": {"disturbance_count": 0.0, "recovery_success_count": 0.0, "controllable_locomotion": 0.9},
    },
    {
      "type": "step",
      "step": 2,
      "command": {"tracking_rate": 0.85},
      "root": {"fallen_rate": 0.0},
      "metrics": {"disturbance_count": 0.0, "recovery_success_count": 0.0, "controllable_locomotion": 0.85},
    },
    {
      "type": "step",
      "step": 3,
      "command": {"tracking_rate": 0.1},
      "root": {"fallen_rate": 1.0},
      "metrics": {"disturbance_count": 1.0, "recovery_success_count": 0.0, "controllable_locomotion": 0.1},
    },
    {
      "type": "step",
      "step": 4,
      "command": {"tracking_rate": 0.3},
      "root": {"fallen_rate": 1.0},
      "metrics": {"disturbance_count": 0.0, "recovery_success_count": 0.0, "controllable_locomotion": 0.0},
    },
  ]

  summary = diag.summarize_records(records, success_threshold=0.8)

  assert summary["disturbance_boundary_step"] == 3
  assert summary["pre_disturbance_tracking_rate"] == pytest.approx(0.9)
  assert summary["post_disturbance_tracking_rate"] == pytest.approx(0.3)
  assert summary["walk_disturb_recover_resume_gate"] is False


def test_antifall_getup_forced_fall_options_are_recorded_in_metadata() -> None:
  args = diag.build_parser().parse_args(
    [
      "--agent",
      "zero",
      "--force-fall-step",
      "25",
      "--force-fall-prob",
      "0.75",
      "--force-fall-command-quiet-s",
      "2.5",
      "--seed",
      "104",
      "--disable-interval-push",
    ]
  )

  metadata = diag.build_metadata_record(args, num_envs=4, clip_actions=2.0)

  assert metadata["forced_fall_step"] == 25
  assert metadata["forced_fall_prob"] == pytest.approx(0.75)
  assert metadata["forced_fall_command_quiet_s"] == pytest.approx(2.5)
  assert metadata["seed"] == 104
  assert metadata["disable_interval_push"] is True


def test_antifall_getup_disable_interval_push_removes_push_event() -> None:
  args = diag.build_parser().parse_args(["--agent", "zero", "--disable-interval-push"])
  push_event = object()
  cfg = SimpleNamespace(events={"push_robot": push_event, "randomize_terrain": object()})

  diag.apply_diagnostic_overrides(cfg, args)

  assert "push_robot" not in cfg.events
  assert "randomize_terrain" in cfg.events




def test_antifall_getup_seed_option_configures_torch_rng(monkeypatch) -> None:
  import torch

  calls = []
  monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(int(seed)))
  args = diag.build_parser().parse_args(["--agent", "zero", "--seed", "104"])

  diag.configure_rollout_seed(args)

  assert calls == [104]

def test_antifall_getup_force_fall_reset_marks_near_failure_disturbance(monkeypatch) -> None:
  root_calls = []
  joint_calls = []
  action_reset_calls = []
  quiet_calls = []

  def fake_reset_root_state_from_presets(env, env_ids, **kwargs):
    root_calls.append((env, env_ids, kwargs))

  def fake_reset_joints_from_presets(env, env_ids, **kwargs):
    joint_calls.append((env, env_ids, kwargs))

  monkeypatch.setattr(diag.mdp, "reset_root_state_from_presets", fake_reset_root_state_from_presets)
  monkeypatch.setattr(diag.mdp, "reset_joints_from_presets", fake_reset_joints_from_presets)
  monkeypatch.setattr(
    diag.mdp,
    "quiet_velocity_command_for_recovery",
    lambda env, env_ids, **kwargs: quiet_calls.append((env, env_ids.clone(), kwargs)),
  )
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    action_manager=SimpleNamespace(reset=lambda env_ids=None: action_reset_calls.append(env_ids.clone())),
  )

  diag.force_fall_reset(env, prob=1.0, command_quiet_s=2.5)

  assert len(root_calls) == 1
  assert len(joint_calls) == 1
  _, env_ids, kwargs = root_calls[0]
  assert env_ids.tolist() == [0, 1, 2]
  assert kwargs["velocity_range"] == diag.FORCED_FALL_VELOCITY_RANGE
  assert kwargs["presets"][1]["pose_range"]["z"] == (-0.55, -0.45)
  assert kwargs["preset_weight_stages"][0]["weights"][-1] > 0.0
  _, joint_env_ids, joint_kwargs = joint_calls[0]
  assert joint_env_ids.tolist() == [0, 1, 2]
  assert joint_kwargs["position_noise_range"] == (-0.05, 0.05)
  assert joint_kwargs["velocity_range"] == (-0.5, 0.5)
  assert len(action_reset_calls) == 1
  assert action_reset_calls[0].tolist() == [0, 1, 2]
  assert len(quiet_calls) == 1
  _, quiet_env_ids, quiet_kwargs = quiet_calls[0]
  assert quiet_env_ids.tolist() == [0, 1, 2]
  assert quiet_kwargs["command_name"] == "twist"
  assert quiet_kwargs["quiet_s"] == pytest.approx(2.5)


def test_antifall_getup_step_record_includes_reward_term_telemetry() -> None:
  import torch

  reward_manager = SimpleNamespace(
    _term_names=["host_lift_progress", "body_orientation_l2"],
    _step_reward=torch.tensor([[1.0, -0.5], [0.5, -1.5]]),
  )
  env = SimpleNamespace(reward_manager=reward_manager)

  terms = diag._reward_terms(env)

  assert terms["host_lift_progress"] == pytest.approx(0.75)
  assert terms["body_orientation_l2"] == pytest.approx(-1.0)


def test_antifall_getup_step_record_includes_action_and_target_telemetry() -> None:
  import torch

  asset = SimpleNamespace(
    body_names=("torso_link",),
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.4]]),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.7]], [[0.0, 0.0, 0.3]]]),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.8, 0.0, -0.6]]),
      root_link_lin_vel_b=torch.zeros(2, 3),
      root_link_ang_vel_b=torch.zeros(2, 3),
      joint_pos=torch.zeros(2, 2),
    ),
  )
  term = SimpleNamespace(
    _raw_actions=torch.tensor([[1.5, -0.5], [0.25, -0.75]]),
    _processed_actions=torch.tensor([[0.8, -0.4], [0.1, -0.6]]),
    _recovery_phase_active=torch.tensor([True, False]),
  )
  env = SimpleNamespace(
    scene={"robot": asset},
    command_manager=None,
    metrics_manager=None,
    reward_manager=None,
    action_manager=SimpleNamespace(_terms={"joint_pos": term}),
    _host_getup_joint_position_target=torch.tensor([[0.9, -0.3], [0.2, -0.7]]),
    _host_getup_joint_position_delta=torch.tensor([[0.4, -0.1], [0.05, -0.35]]),
  )

  record = diag.build_step_record(
    env,
    step_index=7,
    raw_action=torch.tensor([[2.0, -0.5], [0.25, -1.5]]),
    clipped_action=torch.tensor([[1.0, -0.5], [0.25, -1.0]]),
    previous_clipped_action=torch.tensor([[0.5, -0.25], [0.25, -0.25]]),
    clip_actions=1.0,
    rewards=torch.zeros(2),
    dones=torch.zeros(2, dtype=torch.bool),
    extras={},
  )

  assert record["action"]["clip_actions"] == pytest.approx(1.0)
  assert record["action"]["raw_max_abs"] == pytest.approx(2.0)
  assert record["action"]["clipped_max_abs"] == pytest.approx(1.0)
  assert record["action"]["action_rate_max_abs"] == pytest.approx(0.75)
  assert record["action"]["term_raw_max_abs"] == pytest.approx(1.5)
  assert record["action"]["processed_max_abs"] == pytest.approx(0.8)
  assert record["action"]["recovery_phase_active_rate"] == pytest.approx(0.5)
  assert record["action_phase"]["recovery_phase_active_rate"] == pytest.approx(0.5)
  assert record["action_phase"]["coarse_stable_rate"] == pytest.approx(0.5)
  assert record["action_phase"]["stable_exit_ready_rate"] == pytest.approx(0.5)
  assert record["target"]["joint_target_delta_max"] == pytest.approx(0.4)
  assert record["target"]["joint_target_abs_max"] == pytest.approx(0.9)




def test_antifall_getup_step_record_can_include_per_env_command_phase_trace() -> None:
  import torch

  asset = SimpleNamespace(
    body_names=("torso_link",),
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.4]]),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.7]], [[0.0, 0.0, 0.3]]]),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.8, 0.0, -0.6]]),
      root_link_lin_vel_b=torch.tensor([[0.2, 0.1, 0.0], [0.0, 0.0, 0.0]]),
      root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.1]]),
      joint_pos=torch.zeros(2, 2),
    ),
  )
  term = SimpleNamespace(
    _raw_actions=torch.zeros(2, 2),
    _processed_actions=torch.zeros(2, 2),
    _recovery_phase_active=torch.tensor([True, False]),
  )
  command = torch.tensor([[0.3, 0.1, 0.2], [0.6, 0.0, 0.4]])
  env = SimpleNamespace(
    scene={"robot": asset},
    command_manager=SimpleNamespace(get_command=lambda name: command),
    metrics_manager=None,
    reward_manager=None,
    action_manager=SimpleNamespace(_terms={"joint_pos": term}),
  )

  record = diag.build_step_record(
    env,
    step_index=9,
    raw_action=torch.zeros(2, 2),
    clipped_action=torch.zeros(2, 2),
    previous_clipped_action=None,
    clip_actions=None,
    rewards=torch.zeros(2),
    dones=torch.zeros(2, dtype=torch.bool),
    extras={},
    include_env_trace=True,
  )

  assert record["env_trace"]["fallen"] == [False, True]
  assert record["env_trace"]["recovery_phase_active"] == [True, False]
  assert record["env_trace"]["tracking"] == [True, False]
  assert record["env_trace"]["command"] == [[0.3, 0.1, 0.2], [0.6, 0.0, 0.4]]

def test_antifall_getup_step_record_includes_assist_telemetry() -> None:
  import torch

  asset = SimpleNamespace(
    body_names=("torso_link",),
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.4]]),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.3]]]),
      projected_gravity_b=torch.tensor([[0.8, 0.0, -0.6]]),
      root_link_lin_vel_b=torch.zeros(1, 3),
      root_link_ang_vel_b=torch.zeros(1, 3),
      joint_pos=torch.zeros(1, 2),
    ),
  )
  env = SimpleNamespace(
    scene={"robot": asset},
    command_manager=None,
    metrics_manager=None,
    reward_manager=None,
    action_manager=SimpleNamespace(_terms={}),
    _host_getup_latest_assist={
      "active_rate": torch.tensor(0.75),
      "phase_active_rate": torch.tensor(1.0),
      "episode_force_scale_mean": torch.tensor(0.9),
      "assist_fraction_mean": torch.tensor(0.8),
      "force_z_mean": torch.tensor(64.0),
      "force_z_max": torch.tensor(100.0),
    },
  )

  record = diag.build_step_record(
    env,
    step_index=3,
    raw_action=torch.zeros(1, 2),
    clipped_action=torch.zeros(1, 2),
    previous_clipped_action=None,
    clip_actions=None,
    rewards=torch.zeros(1),
    dones=torch.zeros(1, dtype=torch.bool),
    extras={},
  )

  assert record["assist"]["active_rate"] == pytest.approx(0.75)
  assert record["assist"]["phase_active_rate"] == pytest.approx(1.0)
  assert record["assist"]["episode_force_scale_mean"] == pytest.approx(0.9)
  assert record["assist"]["assist_fraction_mean"] == pytest.approx(0.8)
  assert record["assist"]["force_z_mean"] == pytest.approx(64.0)
  assert record["assist"]["force_z_max"] == pytest.approx(100.0)
