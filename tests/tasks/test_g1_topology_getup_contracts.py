from types import SimpleNamespace

import mjlab.tasks  # noqa: F401
import numpy as np
import pytest
import src.tasks  # noqa: F401
import torch
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.anti_fall.events import (
  DISTURBANCE_NEAR_FAILURE_RESET,
  disturbance_window_mask,
  get_antifall_state,
)
from src.tasks.velocity.mdp.topology_getup import events as topology_getup_events
from src.tasks.velocity.mdp.topology_getup import observations as topology_getup_observations
from src.tasks.velocity.mdp.topology_getup import rewards as topology_getup_rewards

_EXPECTED_ANTIFALL_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)

_EXPECTED_TOPOLOGY_ACTOR_TERMS = (
  *_EXPECTED_ANTIFALL_ACTOR_TERMS,
  "getup_progress",
)


class _MockAssetCfg:
  def __init__(self, body_ids: int = 0) -> None:
    self.name = "robot"
    self.body_ids = torch.tensor([body_ids])


def _mock_env(*, projected_gravity_b: tuple[float, float, float], torso_height: float):
  asset = SimpleNamespace(
    data=SimpleNamespace(
      projected_gravity_b=torch.tensor([projected_gravity_b], dtype=torch.float32),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, torso_height]]], dtype=torch.float32),
    )
  )
  return SimpleNamespace(scene={"robot": asset})


def test_stage0_keeps_existing_antifall_actor_contract_but_adds_camera_group() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  assert tuple(cfg.observations["actor"].terms) == _EXPECTED_TOPOLOGY_ACTOR_TERMS
  assert "camera" in cfg.observations
  assert tuple(cfg.observations["camera"].terms) == ("support_depth",)
  assert "getup_progress" in cfg.observations["critic"].terms
  assert "support_contact_pattern" in cfg.observations["critic"].terms
  assert "support_body_contact_count" in cfg.metrics
  assert "torso_clearance" in cfg.metrics
  assert "getup_posture_reward" in cfg.rewards
  assert "getup_torso_lift_reward" in cfg.rewards
  assert "getup_facing_up_reward" in cfg.rewards
  assert "getup_orientation_phase_bonus" in cfg.rewards
  assert "getup_height_progress_reward" in cfg.rewards
  assert "getup_phase_bonus" in cfg.rewards
  assert "support_contact_diversity_reward" in cfg.rewards
  assert cfg.rewards["support_contact_diversity_reward"].params["active_below_height"] == 0.2
  assert "support_body_contact_penalty_after_lift" in cfg.rewards
  assert cfg.rewards["support_body_contact_penalty_after_lift"].params["activation_height"] == 0.2
  assert "getup_standing_joint_pose_reward" in cfg.rewards
  assert "getup_demo_pose_reward" in cfg.rewards
  assert "reduced_support_bonus" in cfg.rewards
  assert cfg.rewards["action_rate_l2"].func is mdp.action_rate_after_lift
  assert cfg.rewards["track_linear_velocity"].func is mdp.track_linear_velocity_after_lift
  assert cfg.rewards["track_angular_velocity"].func is mdp.track_angular_velocity_after_lift
  assert cfg.rewards["joint_pos_limits"].func is mdp.joint_pos_limits_after_support
  assert cfg.rewards["self_collisions"].func is mdp.self_collision_cost_after_support
  assert "pelvis_clearance_penalty" in cfg.rewards
  assert "getup_completion_bonus" in cfg.rewards
  assert "getup_upright" in cfg.metrics
  assert "getup_success_count" in cfg.metrics
  assert "getup_latency" in cfg.metrics
  assert "pelvis_clearance_violation" in cfg.metrics
  assert any(
    isinstance(sensor, CameraSensorCfg) and sensor.name == "support_depth"
    for sensor in (cfg.scene.sensors or ())
  )
  assert cfg.scene.terrain is not None
  assert "is_terminated" not in cfg.rewards
  assert "fell_over" not in cfg.terminations
  assert "head_contact" in cfg.terminations
  assert cfg.terminations["head_contact"].func is mdp.tolerant_illegal_contact
  assert cfg.terminations["head_contact"].params["grace_period_s"] == 1.2
  assert cfg.terminations["head_contact"].params["bad_contact_time_threshold_s"] == 0.5
  assert cfg.terminations["stalled_getup"].func is mdp.stalled_getup_progress
  assert "getup_assist_force" in cfg.events
  assert cfg.events["getup_assist_force"].mode == "step"
  assert cfg.scene.terrain.terrain_type == "generator"
  assert cfg.events["reset_base"].func is mdp.reset_root_state_from_presets
  assert cfg.events["reset_robot_joints"].func is mdp.reset_joints_from_presets
  presets = cfg.events["reset_base"].params["presets"]
  assert tuple(preset["name"] for preset in presets) == ("supine", "left_side", "right_side", "seated_fall")
  assert cfg.events["reset_base"].params["preset_weight_stages"][0]["weights"] == (0.0, 0.25, 0.25, 0.5)


def test_runner_uses_camera_obs_groups_for_both_actor_and_critic() -> None:
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0")
  assert rl_cfg.obs_groups == {
    "actor": ("actor", "camera"),
    "critic": ("critic", "camera"),
  }


def test_benchmark_disables_randomization_without_mutating_antifall_ids() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Benchmark")
  assert cfg.curriculum == {}
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.events["foot_friction"].params["ranges"] == (1.0, 1.0)
  antifall_cfg = load_env_cfg("Unitree-G1-AntiFall-Stage0")
  assert tuple(antifall_cfg.observations["actor"].terms) == _EXPECTED_ANTIFALL_ACTOR_TERMS


def test_runner_uses_topology_bottleneck_model_class() -> None:
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0")
  assert rl_cfg.actor.class_name == "src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel"
  assert rl_cfg.critic.class_name == "src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel"
  assert rl_cfg.actor.cnn_cfg["bottleneck_dim"] == 64


def test_stage0_uses_seen_training_terrain_mix_and_disables_command_curriculum() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  assert tuple(terrain_generator.sub_terrains) == (
    "flat",
    "pyramid_stairs",
    "hf_pyramid_slope",
    "random_rough",
  )
  assert "command_vel" not in cfg.curriculum
  assert "terrain_levels" in cfg.curriculum


def test_stage0_fallen_presets_lower_root_from_standing_default_frame() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  presets = cfg.events["reset_base"].params["presets"]
  z_ranges = {preset["name"]: preset["pose_range"]["z"] for preset in presets}

  assert z_ranges["supine"][1] < -0.5
  assert z_ranges["left_side"][1] < -0.5
  assert z_ranges["right_side"][1] < -0.5
  assert z_ranges["seated_fall"][1] < -0.3


def test_benchmark_switches_to_holdout_terrain_mix() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Benchmark")
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  assert tuple(terrain_generator.sub_terrains) == (
    "open_stairs",
    "random_stairs",
    "random_spread_boxes",
  )


def test_getup_height_progress_reward_increases_with_height_and_rejects_upside_down() -> None:
  low_env = _mock_env(projected_gravity_b=(0.0, 0.0, -1.0), torso_height=0.18)
  high_env = _mock_env(projected_gravity_b=(0.0, 0.0, -1.0), torso_height=0.52)
  upside_down_env = _mock_env(projected_gravity_b=(0.0, 0.0, 1.0), torso_height=0.52)

  low_reward = topology_getup_rewards.getup_height_progress_reward(low_env, asset_cfg=_MockAssetCfg())
  high_reward = topology_getup_rewards.getup_height_progress_reward(high_env, asset_cfg=_MockAssetCfg())
  upside_down_reward = topology_getup_rewards.getup_height_progress_reward(
    upside_down_env, asset_cfg=_MockAssetCfg()
  )

  assert high_reward.item() > low_reward.item()
  assert upside_down_reward.item() == 0.0


def test_getup_torso_lift_reward_increases_with_height_before_full_upright_success() -> None:
  low_env = _mock_env(projected_gravity_b=(1.0, 0.0, 0.0), torso_height=0.18)
  high_env = _mock_env(projected_gravity_b=(1.0, 0.0, 0.0), torso_height=0.52)

  low_reward = topology_getup_rewards.getup_torso_lift_reward(low_env, asset_cfg=_MockAssetCfg())
  high_reward = topology_getup_rewards.getup_torso_lift_reward(high_env, asset_cfg=_MockAssetCfg())

  assert high_reward.item() > low_reward.item()


def test_getup_facing_up_reward_prefers_face_up_orientation() -> None:
  face_up_env = _mock_env(projected_gravity_b=(0.0, 0.0, -1.0), torso_height=0.18)
  upside_down_env = _mock_env(projected_gravity_b=(0.0, 0.0, 1.0), torso_height=0.18)

  face_up_reward = topology_getup_rewards.getup_facing_up_reward(face_up_env, asset_cfg=_MockAssetCfg())
  upside_down_reward = topology_getup_rewards.getup_facing_up_reward(
    upside_down_env, asset_cfg=_MockAssetCfg()
  )

  assert face_up_reward.item() > upside_down_reward.item()


def test_reset_root_state_from_presets_marks_recovery_window(monkeypatch) -> None:
  def _no_reset(*args, **kwargs):
    return None

  monkeypatch.setattr(topology_getup_events.envs_mdp, "reset_root_state_uniform", _no_reset)

  env = SimpleNamespace(num_envs=4, device="cpu", common_step_counter=17, step_dt=0.02)
  env_ids = torch.arange(env.num_envs, dtype=torch.long)

  topology_getup_events.reset_root_state_from_presets(env, env_ids)

  state = get_antifall_state(env)
  assert torch.equal(state["disturbance_count"], torch.ones(env.num_envs, dtype=torch.long))
  assert torch.equal(
    state["disturbance_kind"],
    torch.full((env.num_envs,), DISTURBANCE_NEAR_FAILURE_RESET, dtype=torch.long),
  )
  assert disturbance_window_mask(env, 2.0).all()


def test_preset_weight_schedule_selects_stage_by_common_step_counter() -> None:
  stages = [
    {"step": 0, "weights": (0.0, 0.25, 0.25, 0.5)},
    {"step": 100, "weights": (0.15, 0.25, 0.25, 0.35)},
    {"step": 200, "weights": (0.25, 0.25, 0.25, 0.25)},
  ]
  assert topology_getup_events._preset_weights_for_step(0, stages) == (0.0, 0.25, 0.25, 0.5)
  assert topology_getup_events._preset_weights_for_step(150, stages) == (0.15, 0.25, 0.25, 0.35)
  assert topology_getup_events._preset_weights_for_step(250, stages) == (0.25, 0.25, 0.25, 0.25)


def test_reset_root_state_from_presets_uses_curriculum_weights(monkeypatch) -> None:
  captured = {}

  def _no_reset(*args, **kwargs):
    return None

  def _record_multinomial(weights, num_samples, replacement=True):
    captured["weights"] = tuple(float(x) for x in weights.tolist())
    return torch.zeros(num_samples, dtype=torch.long, device=weights.device)

  monkeypatch.setattr(topology_getup_events.envs_mdp, "reset_root_state_uniform", _no_reset)
  monkeypatch.setattr(topology_getup_events.torch, "multinomial", _record_multinomial)

  env = SimpleNamespace(num_envs=4, device="cpu", common_step_counter=150, step_dt=0.02)
  env_ids = torch.arange(env.num_envs, dtype=torch.long)
  stages = [
    {"step": 0, "weights": (0.0, 0.25, 0.25, 0.5)},
    {"step": 100, "weights": (0.15, 0.25, 0.25, 0.35)},
  ]

  topology_getup_events.reset_root_state_from_presets(env, env_ids, preset_weight_stages=stages)

  assert captured["weights"] == (0.15000000596046448, 0.25, 0.25, 0.3499999940395355)


def test_reset_root_state_from_presets_records_selected_preset_for_joint_reset(monkeypatch) -> None:
  def _no_reset(*args, **kwargs):
    return None

  monkeypatch.setattr(topology_getup_events.envs_mdp, "reset_root_state_uniform", _no_reset)

  env = SimpleNamespace(num_envs=2, device="cpu", common_step_counter=0, step_dt=0.02)
  env_ids = torch.arange(env.num_envs, dtype=torch.long)

  def _choose_last_preset(weights, num_samples, replacement=True):
    return torch.full((num_samples,), 3, dtype=torch.long, device=weights.device)

  monkeypatch.setattr(topology_getup_events.torch, "multinomial", _choose_last_preset)

  topology_getup_events.reset_root_state_from_presets(
    env,
    env_ids,
    preset_weight_stages=[{"step": 0, "weights": (0.0, 0.0, 0.0, 1.0)}],
  )

  state = topology_getup_events.get_topology_getup_reset_state(env)
  assert state["preset_names"] == ("supine", "left_side", "right_side", "seated_fall")
  assert torch.equal(state["preset_index"], torch.tensor([3, 3], dtype=torch.long))


def test_reset_joints_from_presets_uses_selected_fallen_joint_targets() -> None:
  calls = []
  robot = SimpleNamespace(
    joint_names=[
      "left_hip_pitch_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "right_hip_pitch_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
    ],
    data=SimpleNamespace(
      default_joint_pos=torch.zeros((1, 6), dtype=torch.float32),
      default_joint_vel=torch.zeros((1, 6), dtype=torch.float32),
      soft_joint_pos_limits=torch.tensor(
        [[[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]],
        dtype=torch.float32,
      ),
    ),
    write_joint_state_to_sim=lambda joint_pos, joint_vel, env_ids=None, joint_ids=None: calls.append(
      (joint_pos.clone(), joint_vel.clone(), env_ids, joint_ids)
    ),
  )
  env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot})
  state = topology_getup_events.get_topology_getup_reset_state(env, preset_names=("supine",))
  state["preset_index"][:] = 0

  topology_getup_events.reset_joints_from_presets(
    env,
    torch.tensor([0], dtype=torch.long),
    position_noise_range=(0.0, 0.0),
    velocity_range=(0.0, 0.0),
    preset_joint_targets={
      "supine": {
        "left_hip_pitch_joint": -1.0,
        "left_knee_joint": 1.8,
        "left_ankle_pitch_joint": -0.8,
        "right_hip_pitch_joint": -1.0,
        "right_knee_joint": 1.8,
        "right_ankle_pitch_joint": -0.8,
      }
    },
    asset_cfg=SimpleNamespace(name="robot", joint_ids=slice(None)),
  )

  assert len(calls) == 1
  joint_pos, joint_vel, env_ids, _joint_ids = calls[0]
  assert env_ids.tolist() == [0]
  assert torch.allclose(joint_vel, torch.zeros_like(joint_vel))
  assert torch.allclose(
    joint_pos,
    torch.tensor([[-1.0, 1.8, -0.8, -1.0, 1.8, -0.8]], dtype=torch.float32),
  )


def test_tolerant_illegal_contact_requires_persistent_contact_after_grace_period() -> None:
  sensor = SimpleNamespace(
    data=SimpleNamespace(force_history=torch.zeros((1, 1, 1, 3), dtype=torch.float32), found=None)
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    episode_length_buf=torch.zeros(1, dtype=torch.long),
    scene={"head_sensor": sensor},
  )
  term = mdp.tolerant_illegal_contact(SimpleNamespace(params={}), env)

  sensor.data.force_history[:] = 10.0
  env.episode_length_buf[:] = 1
  assert not term(
    env,
    sensor_name="head_sensor",
    force_threshold=5.0,
    grace_period_s=0.04,
    bad_contact_time_threshold_s=0.06,
  ).item()

  env.episode_length_buf[:] = 2
  assert not term(
    env,
    sensor_name="head_sensor",
    force_threshold=5.0,
    grace_period_s=0.04,
    bad_contact_time_threshold_s=0.06,
  ).item()

  env.episode_length_buf[:] = 3
  assert not term(
    env,
    sensor_name="head_sensor",
    force_threshold=5.0,
    grace_period_s=0.04,
    bad_contact_time_threshold_s=0.06,
  ).item()

  env.episode_length_buf[:] = 4
  assert term(
    env,
    sensor_name="head_sensor",
    force_threshold=5.0,
    grace_period_s=0.04,
    bad_contact_time_threshold_s=0.06,
  ).item()


def test_getup_phase_bonus_pays_once_per_height_stage_and_rearms_on_reset() -> None:
  env = _mock_env(projected_gravity_b=(0.0, 0.0, -1.0), torso_height=0.15)
  env.num_envs = 1
  env.device = "cpu"
  bonus = topology_getup_rewards.getup_phase_bonus(SimpleNamespace(params={}), env)

  assert bonus(env, asset_cfg=_MockAssetCfg(), thresholds=(0.2, 0.4), bonuses=(1.0, 2.0)).item() == 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.25]]], dtype=torch.float32)
  first = bonus(env, asset_cfg=_MockAssetCfg(), thresholds=(0.2, 0.4), bonuses=(1.0, 2.0)).item()
  assert first == 1.0
  assert bonus(env, asset_cfg=_MockAssetCfg(), thresholds=(0.2, 0.4), bonuses=(1.0, 2.0)).item() == 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.45]]], dtype=torch.float32)
  second = bonus(env, asset_cfg=_MockAssetCfg(), thresholds=(0.2, 0.4), bonuses=(1.0, 2.0)).item()
  assert second == 2.0

  bonus.reset()
  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.25]]], dtype=torch.float32)
  assert bonus(env, asset_cfg=_MockAssetCfg(), thresholds=(0.2, 0.4), bonuses=(1.0, 2.0)).item() == 1.0


def test_getup_orientation_phase_bonus_pays_once_per_alignment_stage() -> None:
  env = _mock_env(projected_gravity_b=(0.0, 0.0, 1.0), torso_height=0.15)
  env.num_envs = 1
  env.device = "cpu"
  bonus = topology_getup_rewards.getup_orientation_phase_bonus(SimpleNamespace(params={}), env)

  assert bonus(env, thresholds=(0.1, 0.5), bonuses=(1.0, 2.0), asset_cfg=_MockAssetCfg()).item() == 0.0

  env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0.0, 0.0, -0.2]], dtype=torch.float32)
  assert bonus(env, thresholds=(0.1, 0.5), bonuses=(1.0, 2.0), asset_cfg=_MockAssetCfg()).item() == 1.0
  assert bonus(env, thresholds=(0.1, 0.5), bonuses=(1.0, 2.0), asset_cfg=_MockAssetCfg()).item() == 0.0

  env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0.0, 0.0, -0.8]], dtype=torch.float32)
  assert bonus(env, thresholds=(0.1, 0.5), bonuses=(1.0, 2.0), asset_cfg=_MockAssetCfg()).item() == 2.0

  bonus.reset()
  env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0.0, 0.0, -0.2]], dtype=torch.float32)
  assert bonus(env, thresholds=(0.1, 0.5), bonuses=(1.0, 2.0), asset_cfg=_MockAssetCfg()).item() == 1.0


def test_stand_still_after_getup_only_penalizes_after_lift_and_facing_up() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.tensor([[0.3, -0.2]], dtype=torch.float32),
      default_joint_pos=torch.zeros((1, 2), dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.2]]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    scene={"robot": robot},
    command_manager=SimpleNamespace(get_command=lambda name: torch.zeros((1, 3), dtype=torch.float32)),
  )
  joint_asset_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0, 1]))

  low_penalty = topology_getup_rewards.stand_still_after_getup(
    env,
    command_name="twist",
    joint_asset_cfg=joint_asset_cfg,
    torso_asset_cfg=_MockAssetCfg(),
    activation_height=0.45,
  )
  assert low_penalty.item() == 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.55]]], dtype=torch.float32)
  high_penalty = topology_getup_rewards.stand_still_after_getup(
    env,
    command_name="twist",
    joint_asset_cfg=joint_asset_cfg,
    torso_asset_cfg=_MockAssetCfg(),
    activation_height=0.45,
  )
  assert high_penalty.item() > 0.0


def test_support_contact_diversity_reward_only_applies_before_lift() -> None:
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 1.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.2]]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(scene={"support_sensor": sensor, "robot": robot})

  early = topology_getup_rewards.support_contact_diversity_reward(
    env,
    sensor_name="support_sensor",
    target_count=2.0,
    active_below_height=0.35,
    asset_cfg=_MockAssetCfg(),
  )
  assert early.item() > 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.5]]], dtype=torch.float32)
  late = topology_getup_rewards.support_contact_diversity_reward(
    env,
    sensor_name="support_sensor",
    target_count=2.0,
    active_below_height=0.35,
    asset_cfg=_MockAssetCfg(),
  )
  assert late.item() == 0.0


def test_getup_progress_features_reflect_height_alignment_body_and_feet_support() -> None:
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 0.0]], dtype=torch.float32)))
  feet_sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 1.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.4]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -0.8]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(scene={"support_sensor": sensor, "feet_sensor": feet_sensor, "robot": robot})

  features = topology_getup_observations.getup_progress_features(
    env,
    sensor_name="support_sensor",
    feet_sensor_name="feet_sensor",
    asset_cfg=_MockAssetCfg(),
  )

  assert features.shape == (1, 4)
  assert features[0, 0].item() > 0.0
  assert features[0, 1].item() > 0.0
  assert features[0, 2].item() > 0.0
  assert features[0, 3].item() > 0.0


def test_getup_progress_uses_height_relative_to_env_origin_when_available() -> None:
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0, 0.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 1.2]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    scene={
      "support_sensor": sensor,
      "robot": robot,
      "env_origins": torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    }
  )

  features = topology_getup_observations.getup_progress_features(
    env,
    sensor_name="support_sensor",
    min_height=0.12,
    target_height=0.55,
    asset_cfg=_MockAssetCfg(),
  )

  expected_height_progress = (0.2 - 0.12) / (0.55 - 0.12)
  assert features[0, 0].item() == pytest.approx(expected_height_progress)


def test_support_body_contact_penalty_after_lift_only_activates_after_lift() -> None:
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 1.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.2]]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(scene={"support_sensor": sensor, "robot": robot})

  low = topology_getup_rewards.support_body_contact_penalty_after_lift(
    env,
    sensor_name="support_sensor",
    activation_height=0.35,
    asset_cfg=_MockAssetCfg(),
  )
  assert low.item() == 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.5]]], dtype=torch.float32)
  high = topology_getup_rewards.support_body_contact_penalty_after_lift(
    env,
    sensor_name="support_sensor",
    activation_height=0.35,
    asset_cfg=_MockAssetCfg(),
  )
  assert high.item() > 0.0


def test_getup_standing_joint_pose_reward_requires_feet_support_and_reduced_body_support() -> None:
  feet_sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 0.0]], dtype=torch.float32)))
  body_sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0, 0.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    joint_names=[
      "left_hip_pitch_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "right_hip_pitch_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
    ],
    data=SimpleNamespace(
      joint_pos=torch.tensor([[-1.0, 1.6, -0.7, -1.0, 1.6, -0.7]], dtype=torch.float32),
      default_joint_pos=torch.tensor([[-0.1, 0.3, -0.2, -0.1, 0.3, -0.2]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    ),
  )
  env = SimpleNamespace(scene={"feet_sensor": feet_sensor, "body_sensor": body_sensor, "robot": robot})

  active = topology_getup_rewards.getup_standing_joint_pose_reward(
    env,
    feet_sensor_name="feet_sensor",
    body_sensor_name="body_sensor",
    joint_names=tuple(robot.joint_names),
  )
  assert active.item() > 0.0

  env.scene["feet_sensor"].data.found[:] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
  inactive_without_feet = topology_getup_rewards.getup_standing_joint_pose_reward(
    env,
    feet_sensor_name="feet_sensor",
    body_sensor_name="body_sensor",
    joint_names=tuple(robot.joint_names),
  )
  assert inactive_without_feet.item() == 0.0

  env.scene["feet_sensor"].data.found[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
  env.scene["body_sensor"].data.found = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
  inactive_with_heavy_body_support = topology_getup_rewards.getup_standing_joint_pose_reward(
    env,
    feet_sensor_name="feet_sensor",
    body_sensor_name="body_sensor",
    joint_names=tuple(robot.joint_names),
    max_body_support_count=1.0,
  )
  assert inactive_with_heavy_body_support.item() == 0.0


def test_getup_demo_pose_reward_tracks_sequence_and_resets(tmp_path) -> None:
  demo_path = tmp_path / "demo.npz"
  np.savez(
    demo_path,
    joint_pos=np.array(
      [
        [0.0, 0.0],
        [0.5, 0.5],
      ],
      dtype=np.float32,
    ),
  )
  robot = SimpleNamespace(
    joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"],
    data=SimpleNamespace(joint_pos=torch.tensor([[0.0, 0.0]], dtype=torch.float32)),
  )
  env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot})
  reward = topology_getup_rewards.getup_demo_pose_reward(
    SimpleNamespace(params={}),
    env,
    demo_npz_path=str(demo_path),
    joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
    dt_per_demo_frame=0.02,
  )

  first = reward(
    env,
    demo_npz_path=str(demo_path),
    joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
    dt_per_demo_frame=0.02,
  )
  assert first.item() > 0.9

  env.scene["robot"].data.joint_pos[:] = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
  second = reward(
    env,
    demo_npz_path=str(demo_path),
    joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
    dt_per_demo_frame=0.02,
  )
  assert second.item() > 0.9

  reward.reset()
  env.scene["robot"].data.joint_pos[:] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
  reset_val = reward(
    env,
    demo_npz_path=str(demo_path),
    joint_names=("left_hip_pitch_joint", "right_hip_pitch_joint"),
    dt_per_demo_frame=0.02,
  )
  assert reset_val.item() > 0.9


def test_joint_pos_limits_after_support_requires_feet_support() -> None:
  feet_sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0, 0.0]], dtype=torch.float32)))
  body_sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[0.0, 0.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=torch.tensor([[1.5]], dtype=torch.float32),
      soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(scene={"feet_sensor": feet_sensor, "body_sensor": body_sensor, "robot": robot})
  asset_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0]))

  inactive = topology_getup_rewards.joint_pos_limits_after_support(
    env,
    feet_sensor_name="feet_sensor",
    body_sensor_name="body_sensor",
    asset_cfg=asset_cfg,
  )
  assert inactive.item() == 0.0

  env.scene["feet_sensor"].data.found[:] = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
  active = topology_getup_rewards.joint_pos_limits_after_support(
    env,
    feet_sensor_name="feet_sensor",
    body_sensor_name="body_sensor",
    asset_cfg=asset_cfg,
  )
  assert active.item() > 0.0


def test_reduced_support_bonus_pays_once_when_supports_drop_after_progress() -> None:
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1.0, 1.0]], dtype=torch.float32)))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.6]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(num_envs=1, device='cpu', scene={"support_sensor": sensor, "robot": robot})
  bonus = topology_getup_rewards.reduced_support_bonus(SimpleNamespace(params={}), env)

  assert bonus(
    env,
    sensor_name="support_sensor",
    max_support_count=0.5,
    activation_height=0.4,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  ).item() == 0.0

  env.scene["support_sensor"].data.found[:] = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
  first = bonus(
    env,
    sensor_name="support_sensor",
    max_support_count=0.5,
    activation_height=0.4,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  ).item()
  assert first == 1.0
  assert bonus(
    env,
    sensor_name="support_sensor",
    max_support_count=0.5,
    activation_height=0.4,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  ).item() == 0.0

  bonus.reset()
  assert bonus(
    env,
    sensor_name="support_sensor",
    max_support_count=0.5,
    activation_height=0.4,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  ).item() == 1.0


def test_action_rate_after_lift_only_penalizes_after_lift() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.15]]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    scene={"robot": robot},
    action_manager=SimpleNamespace(
      action=torch.tensor([[0.5, -0.5]], dtype=torch.float32),
      prev_action=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
    ),
    last_action=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
  )

  low = topology_getup_rewards.action_rate_after_lift(
    env,
    activation_height=0.25,
    asset_cfg=_MockAssetCfg(),
  )
  assert low.item() == 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.35]]], dtype=torch.float32)
  high = topology_getup_rewards.action_rate_after_lift(
    env,
    activation_height=0.25,
    asset_cfg=_MockAssetCfg(),
  )
  assert high.item() > 0.0


def test_tracking_rewards_only_activate_after_lift_and_facing_up() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.2]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
      root_link_lin_vel_b=torch.zeros((1, 3), dtype=torch.float32),
      root_link_ang_vel_b=torch.zeros((1, 3), dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    scene={"robot": robot},
    command_manager=SimpleNamespace(get_command=lambda name: torch.zeros((1, 3), dtype=torch.float32)),
  )

  low_linear = topology_getup_rewards.track_linear_velocity_after_lift(
    env,
    std=0.5,
    command_name="twist",
    activation_height=0.45,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  )
  low_angular = topology_getup_rewards.track_angular_velocity_after_lift(
    env,
    std=0.5,
    command_name="twist",
    activation_height=0.45,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  )
  assert low_linear.item() == 0.0
  assert low_angular.item() == 0.0

  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.6]]], dtype=torch.float32)
  high_linear = topology_getup_rewards.track_linear_velocity_after_lift(
    env,
    std=0.5,
    command_name="twist",
    activation_height=0.45,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  )
  high_angular = topology_getup_rewards.track_angular_velocity_after_lift(
    env,
    std=0.5,
    command_name="twist",
    activation_height=0.45,
    alignment_threshold=0.3,
    asset_cfg=_MockAssetCfg(),
  )
  assert high_linear.item() > 0.0
  assert high_angular.item() > 0.0


def test_stalled_getup_progress_terminates_only_after_grace_without_progress() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.12]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={"robot": robot},
    episode_length_buf=torch.tensor([20], dtype=torch.long),
    step_dt=0.02,
  )
  term = mdp.stalled_getup_progress(SimpleNamespace(params={}), env)

  early = term(
    env,
    min_steps_before_check=50,
    progress_threshold=0.2,
    asset_cfg=_MockAssetCfg(),
  )
  assert early.item() is False

  for _ in range(49):
    stalled = term(
      env,
      min_steps_before_check=50,
      progress_threshold=0.2,
      asset_cfg=_MockAssetCfg(),
    )
  assert stalled.item() is True

  term.reset()
  env.scene["robot"].data.body_link_pos_w[:] = torch.tensor([[[0.0, 0.0, 0.5]]], dtype=torch.float32)
  env.scene["robot"].data.projected_gravity_b[:] = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
  env.episode_length_buf[:] = 80
  progressed = term(
    env,
    min_steps_before_check=50,
    progress_threshold=0.2,
    asset_cfg=_MockAssetCfg(),
  )
  assert progressed.item() is False


def test_stalled_getup_progress_uses_internal_elapsed_steps_not_episode_length_buf() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.12]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene={"robot": robot},
    episode_length_buf=torch.tensor([1000], dtype=torch.long),
    step_dt=0.02,
  )
  term = mdp.stalled_getup_progress(SimpleNamespace(params={}), env)
  term.reset()

  first = term(
    env,
    min_steps_before_check=5,
    progress_threshold=0.2,
    asset_cfg=_MockAssetCfg(),
  )
  assert first.item() is False

  for _ in range(4):
    stalled = term(
      env,
      min_steps_before_check=5,
      progress_threshold=0.2,
      asset_cfg=_MockAssetCfg(),
    )
  assert stalled.item() is True


def test_apply_getup_assist_force_only_pushes_low_non_upside_down_torso() -> None:
  calls = []

  robot = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.15]], [[0.0, 0.0, 0.45]], [[0.0, 0.0, 0.15]]], dtype=torch.float32),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    ),
    write_external_wrench_to_sim=lambda forces, torques, env_ids=None, body_ids=None: calls.append(
      (forces.clone(), torques.clone(), env_ids, body_ids)
    ),
  )
  env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": robot})
  asset_cfg = SimpleNamespace(name="robot", body_ids=[0])
  term = topology_getup_events.apply_getup_assist_force(
    SimpleNamespace(params={"asset_cfg": asset_cfg}), env
  )

  term(
    env,
    None,
    force_n=50.0,
    activation_height=0.3,
    alignment_threshold=0.0,
    asset_cfg=asset_cfg,
  )

  assert len(calls) == 1
  forces, torques, env_ids, body_ids = calls[0]
  assert env_ids.tolist() == [0, 1, 2]
  assert body_ids == [0]
  assert forces.shape == (3, 1, 3)
  assert torques.shape == (3, 1, 3)
  assert forces[0, 0, 2].item() == 50.0
  assert forces[1, 0, 2].item() == 0.0
  assert forces[2, 0, 2].item() == 0.0
