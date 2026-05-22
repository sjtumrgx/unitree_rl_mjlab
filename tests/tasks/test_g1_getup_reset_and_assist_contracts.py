from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.config.g1_getup.env_cfgs import unitree_g1_getup_env_cfg
from src.tasks.velocity.mdp.getup import events


class _FakeAsset:
  def __init__(self):
    self.joint_names = ["known_joint"]
    self.data = SimpleNamespace(
      default_joint_pos=torch.zeros(1, 1),
      default_joint_vel=torch.zeros(1, 1),
      soft_joint_pos_limits=torch.tensor([[[-10.0, 10.0]]]),
      body_link_pos_w=torch.tensor([[[0.0, 0.0, 1.2]]]),
      projected_gravity_b=torch.tensor([[0.8, 0.0, -0.1]]),
    )
    self.wrench_writes = []

  def write_joint_state_to_sim(self, *args, **kwargs):
    self.joint_write = (args, kwargs)

  def write_external_wrench_to_sim(self, forces, torques, env_ids, body_ids):
    self.wrench_writes.append((forces.clone(), torques.clone(), env_ids, body_ids))


class _FakeContactSensor:
  def __init__(self, found: torch.Tensor):
    self.data = SimpleNamespace(found=found)


class _FakeEnv:
  def __init__(self):
    self.num_envs = 1
    self.device = "cpu"
    self.common_step_counter = 0
    self.episode_length_buf = torch.tensor([31], dtype=torch.long)
    self.scene = {
      "robot": _FakeAsset(),
      "env_origins": torch.zeros(1, 3),
      "feet_ground_contact": _FakeContactSensor(torch.tensor([[[1.0]]])),
      "support_body_contact": _FakeContactSensor(torch.tensor([[[0.0]]])),
    }


def test_reset_joints_from_presets_raises_for_unknown_joint_name() -> None:
  env = _FakeEnv()
  events.get_getup_reset_state(env, preset_names=("supine",))["preset_index"][:] = 0

  with pytest.raises(ValueError, match="unknown.*joint"):
    events.reset_joints_from_presets(
      env,
      torch.tensor([0]),
      position_noise_range=(0.0, 0.0),
      velocity_range=(0.0, 0.0),
      preset_joint_targets={"supine": {"missing_joint": 1.0}},
      asset_cfg=SceneEntityCfg("robot", joint_ids=[0]),
    )


def test_reset_joints_from_presets_raises_when_selected_preset_has_no_targets() -> None:
  env = _FakeEnv()
  events.get_getup_reset_state(env, preset_names=("ghost",))["preset_index"][:] = 0

  with pytest.raises(ValueError, match="no joint targets"):
    events.reset_joints_from_presets(
      env,
      torch.tensor([0]),
      position_noise_range=(0.0, 0.0),
      velocity_range=(0.0, 0.0),
      preset_joint_targets={"supine": {"known_joint": 1.0}},
      asset_cfg=SceneEntityCfg("robot", joint_ids=[0]),
    )


def test_supine_reset_uses_higher_fallen_z_offset_to_avoid_contact_pop() -> None:
  cfg = unitree_g1_getup_env_cfg("ground", play=True)
  presets = {
    preset["name"]: preset["pose_range"]
    for preset in cfg.events["reset_base"].params["presets"]
  }

  assert presets["supine"]["z"] == (-0.35, -0.25)
  assert presets["left_side"]["z"] == (-0.7, -0.6)
  assert presets["right_side"]["z"] == (-0.7, -0.6)
  # The offset is still below the standing default root height; this is a
  # collision-impulse guard, not a hidden standing reset.
  assert presets["supine"]["z"][1] < -0.2


def test_play_randomizes_terrain_before_fallen_root_reset() -> None:
  cfg = unitree_g1_getup_env_cfg("slope", play=True)
  reset_terms = [name for name, term in cfg.events.items() if term.mode == "reset"]

  assert "randomize_terrain" in reset_terms
  assert reset_terms.index("randomize_terrain") < reset_terms.index("reset_base")
  assert reset_terms.index("randomize_terrain") < reset_terms.index("reset_robot_joints")


def test_assist_curriculum_does_not_decay_on_ballistic_height_without_stable_support() -> None:
  env = _FakeEnv()
  cfg = SimpleNamespace(params={"asset_cfg": SceneEntityCfg("robot", body_ids=[0])})
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)
  state["max_torso_height"][:] = 1.2

  assist.reset(torch.tensor([0]))

  assert state["force_n"].item() == 100.0
  assert state["action_rescale"].item() == 1.0



def test_assist_curriculum_decays_on_reachable_support_milestone_before_strict_foot_posture() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.58]]])
  env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.0, 0.0, -0.90]])
  env.scene["feet_ground_contact"] = _FakeContactSensor(torch.tensor([[[1.0], [1.0]]]))
  env.scene["support_body_contact"] = _FakeContactSensor(torch.tensor([[[0.0]]]))
  env.scene["hand_ground_contact"] = _FakeContactSensor(torch.tensor([[[0.0], [0.0]]]))
  env.scene["foot_geom_ground_contact"] = _FakeContactSensor(torch.zeros(1, 14, 1))
  cfg = SimpleNamespace(
    params={
      "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
      "success_height_threshold": 0.55,
      "stable_success_required": True,
      "assist_decay_requires_strict_success": False,
      "upright_alignment_threshold": 0.85,
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "hand_sensor_name": "hand_ground_contact",
      "foot_geom_sensor_name": "foot_geom_ground_contact",
      "min_feet_contact_count": 2.0,
      "max_body_support_count": 0.0,
      "max_hand_contact_count": 0.0,
      "min_foot_flatness": 0.6,
      "min_foot_heading_alignment": 0.6,
      "min_foot_geom_contact_spread": 0.5,
      "force_decay_n": 20.0,
      "action_scale_decay": 0.02,
    }
  )
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)
  state["max_torso_height"][:] = 0.58

  assist.reset(torch.tensor([0]))

  assert state["force_n"].item() == pytest.approx(80.0)
  assert state["action_rescale"].item() == pytest.approx(0.98)


def test_assist_curriculum_decays_when_stable_success_reaches_reachable_height() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]])
  cfg = SimpleNamespace(
    params={
      "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
      "success_height_threshold": 0.75,
      "stable_success_required": True,
      "upright_alignment_threshold": 0.85,
      "force_decay_n": 20.0,
      "action_scale_decay": 0.02,
    }
  )
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)
  state["max_torso_height"][:] = 0.82

  assist.reset(torch.tensor([0]))

  assert state["force_n"].item() == pytest.approx(80.0)
  assert state["action_rescale"].item() == pytest.approx(0.98)


def test_assist_curriculum_decays_from_latched_episode_success_even_if_reset_pose_falls() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.82]]])
  env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]])
  cfg = SimpleNamespace(
    params={
      "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
      "success_height_threshold": 0.75,
      "stable_success_required": True,
      "upright_alignment_threshold": 0.85,
      "force_decay_n": 20.0,
      "action_scale_decay": 0.02,
    }
  )
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)

  assist(
    env,
    None,
    initial_force_n=100.0,
    success_height_threshold=0.75,
    stable_success_required=True,
    upright_alignment_threshold=0.85,
    no_orientation_gate=True,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.2]]])
  env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.8, 0.0, -0.1]])

  assist.reset(torch.tensor([0]))

  assert state["force_n"].item() == pytest.approx(80.0)
  assert state["action_rescale"].item() == pytest.approx(0.98)


def test_assist_force_turns_off_immediately_after_episode_success_latches() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.82]]])
  env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0]])
  cfg = SimpleNamespace(
    params={
      "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
      "success_height_threshold": 0.75,
      "stable_success_required": True,
      "upright_alignment_threshold": 0.85,
    }
  )
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)

  assist(
    env,
    None,
    initial_force_n=100.0,
    success_height_threshold=0.75,
    stable_success_required=True,
    upright_alignment_threshold=0.85,
    no_orientation_gate=True,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  forces, _, env_ids, _ = env.scene["robot"].wrench_writes[-1]
  assert env_ids.tolist() == [0]
  assert state["episode_success"].item() is True
  assert forces[0, 0, 2].item() == pytest.approx(0.0)


def test_assist_force_tapers_out_before_success_height() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.575]]])
  cfg = SimpleNamespace(params={"asset_cfg": SceneEntityCfg("robot", body_ids=[0])})
  assist = events.apply_host_getup_assist_force(cfg, env)

  assist(
    env,
    None,
    initial_force_n=100.0,
    unactuated_timesteps=30,
    no_orientation_gate=True,
    taper_start_height=0.45,
    taper_end_height=0.70,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  forces, _, env_ids, _ = env.scene["robot"].wrench_writes[-1]
  assert env_ids.tolist() == [0]
  assert forces[0, 0, 2].item() == pytest.approx(50.0)


def test_configured_assist_is_zero_by_reported_getup_success_height() -> None:
  cfg = unitree_g1_getup_env_cfg("ground")

  reported_success_height = cfg.metrics["getup_upright"].params["torso_height_threshold"]
  assist_params = cfg.events["getup_assist_force"].params

  assert assist_params["success_height_threshold"] == pytest.approx(reported_success_height)
  assert assist_params["taper_end_height"] <= reported_success_height


def test_configured_assist_uses_gradual_no_assist_curriculum_after_reset_order_fix() -> None:
  cfg = unitree_g1_getup_env_cfg("ground")
  assist_params = cfg.events["getup_assist_force"].params

  assert assist_params["initial_force_n"] == pytest.approx(120.0)
  assert assist_params["action_scale_decay"] == pytest.approx(0.0)
  assert assist_params["min_action_scale"] == pytest.approx(1.0)
  # After the play reset-order fix, keep the empirically stable gradual schedule:
  # early training still gets assisted recovery examples, while late training
  # ramps mostly to no-assist play dynamics instead of immediately dropping
  # the bootstrap and catastrophically forgetting platform recovery.
  assert assist_params["no_assist_probability_initial"] == pytest.approx(0.05)
  assert assist_params["no_assist_probability"] == pytest.approx(0.8)
  assert assist_params["no_assist_ramp_start_progress"] == pytest.approx(0.5)
  assert assist_params["no_assist_ramp_end_progress"] == pytest.approx(1.0)


def test_no_assist_episode_mix_ramps_with_assist_force_decay() -> None:
  force = torch.tensor([120.0, 90.0, 60.0, 30.0, 0.0])

  probability = events._scheduled_no_assist_probability(
    force,
    initial_force_n=120.0,
    min_force_n=0.0,
    initial_probability=0.25,
    max_probability=0.80,
    ramp_start_progress=0.2,
    ramp_end_progress=0.8,
  )

  assert probability[0].item() == pytest.approx(0.25)
  assert probability[1].item() == pytest.approx(0.2958333333)
  assert probability[2].item() == pytest.approx(0.525)
  assert probability[3].item() == pytest.approx(0.7541666667)
  assert probability[4].item() == pytest.approx(0.80)


def test_ground_assist_is_limited_to_sparse_bootstrap_not_primary_solution() -> None:
  cfg = unitree_g1_getup_env_cfg("ground")
  assist_params = cfg.events["getup_assist_force"].params

  assert assist_params["initial_force_n"] <= 120.0
  assert assist_params["taper_start_height"] <= 0.35


def test_assist_force_is_zero_at_taper_end_even_before_success_latches() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.70]]])
  env.scene["robot"].data.projected_gravity_b = torch.tensor([[0.8, 0.0, -0.1]])
  cfg = SimpleNamespace(params={"asset_cfg": SceneEntityCfg("robot", body_ids=[0])})
  assist = events.apply_host_getup_assist_force(cfg, env)

  assist(
    env,
    None,
    initial_force_n=100.0,
    unactuated_timesteps=30,
    no_orientation_gate=True,
    stable_success_required=True,
    success_height_threshold=0.75,
    taper_start_height=0.45,
    taper_end_height=0.70,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  forces, _, env_ids, _ = env.scene["robot"].wrench_writes[-1]
  assert env_ids.tolist() == [0]
  assert forces[0, 0, 2].item() == pytest.approx(0.0)


def test_assist_can_sample_no_assist_episode_without_destroying_curriculum_force() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.20]]])
  cfg = SimpleNamespace(
    params={
      "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
      "no_assist_probability": 1.0,
    }
  )
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)

  assist.reset(torch.tensor([0]))
  assist(
    env,
    None,
    initial_force_n=100.0,
    unactuated_timesteps=30,
    no_orientation_gate=True,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  forces, _, env_ids, _ = env.scene["robot"].wrench_writes[-1]
  assert env_ids.tolist() == [0]
  assert state["force_n"].item() == pytest.approx(100.0)
  assert state["episode_force_scale"].item() == pytest.approx(0.0)
  assert forces[0, 0, 2].item() == pytest.approx(0.0)


def test_assist_force_step_accepts_stable_success_config_params() -> None:
  env = _FakeEnv()
  cfg = SimpleNamespace(params={"asset_cfg": SceneEntityCfg("robot", body_ids=[0])})
  assist = events.apply_host_getup_assist_force(cfg, env)

  assist(
    env,
    None,
    stable_success_required=True,
    upright_alignment_threshold=0.85,
    feet_sensor_name="feet_ground_contact",
    body_sensor_name="support_body_contact",
    min_feet_contact_count=1.0,
    max_body_support_count=1.0,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  forces, _, env_ids, _ = env.scene["robot"].wrench_writes[-1]
  assert env_ids.tolist() == [0]
  assert torch.allclose(forces, torch.zeros_like(forces))


def test_assist_force_can_pull_fallen_postures_after_startup_when_orientation_gate_disabled() -> None:
  env = _FakeEnv()
  env.scene["robot"].data.body_link_pos_w = torch.tensor([[[0.0, 0.0, 0.20]]])
  cfg = SimpleNamespace(params={"asset_cfg": SceneEntityCfg("robot", body_ids=[0])})
  assist = events.apply_host_getup_assist_force(cfg, env)

  assist(
    env,
    None,
    initial_force_n=100.0,
    unactuated_timesteps=30,
    no_orientation_gate=True,
    asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
  )

  forces, _, env_ids, _ = env.scene["robot"].wrench_writes[-1]
  assert env_ids.tolist() == [0]
  assert forces[0, 0, 2].item() == pytest.approx(100.0)
