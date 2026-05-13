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


class _FakeEnv:
  def __init__(self):
    self.num_envs = 1
    self.device = "cpu"
    self.common_step_counter = 0
    self.episode_length_buf = torch.tensor([31], dtype=torch.long)
    self.scene = {"robot": _FakeAsset(), "env_origins": torch.zeros(1, 3)}


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


def test_assist_curriculum_does_not_decay_on_ballistic_height_without_stable_support() -> None:
  env = _FakeEnv()
  cfg = SimpleNamespace(params={"asset_cfg": SceneEntityCfg("robot", body_ids=[0])})
  assist = events.apply_host_getup_assist_force(cfg, env)
  state = events.get_host_getup_curriculum_state(env, initial_force_n=100.0, initial_action_scale=1.0)
  state["max_torso_height"][:] = 1.2

  assist.reset(torch.tensor([0]))

  assert state["force_n"].item() == 100.0
  assert state["action_rescale"].item() == 1.0


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
