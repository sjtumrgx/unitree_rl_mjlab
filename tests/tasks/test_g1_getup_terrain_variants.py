from types import SimpleNamespace

import pytest
import torch

from mjlab.entity.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.config.g1_getup.env_cfgs import (
  GETUP_TERRAIN_VARIANTS,
  HOST_TERRAIN_PARITY,
  unitree_g1_getup_env_cfg,
)
from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg
from src.tasks.velocity.mdp.getup.events import apply_getup_assist_force
from src.tasks.velocity.mdp.getup.rewards import (
  getup_height_progress_reward,
  getup_posture_reward,
)
from src.tasks.velocity.mdp.getup.terminations import stalled_getup_progress


class _Scene(dict):
  def __init__(self, *args, env_origins=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.env_origins = env_origins


class _FakeRobot:
  def __init__(self, *, body_heights, projected_gravity=None):
    body_heights_t = torch.as_tensor(body_heights, dtype=torch.float32)
    if body_heights_t.ndim == 1:
      body_heights_t = body_heights_t[:, None]
    num_envs, num_bodies = body_heights_t.shape
    body_pos = torch.zeros(num_envs, num_bodies, 3, dtype=torch.float32)
    body_pos[..., 2] = body_heights_t
    if projected_gravity is None:
      projected_gravity = torch.tensor([[0.0, 0.0, -1.0]] * num_envs)
    self.data = SimpleNamespace(
      body_link_pos_w=body_pos,
      projected_gravity_b=torch.as_tensor(projected_gravity, dtype=torch.float32),
    )
    self.last_forces = None
    self.last_torques = None
    self.last_env_ids = None
    self.last_body_ids = None

  def write_external_wrench_to_sim(self, forces, torques, *, env_ids, body_ids):
    self.last_forces = forces.clone()
    self.last_torques = torques.clone()
    self.last_env_ids = env_ids.clone() if hasattr(env_ids, "clone") else env_ids
    self.last_body_ids = body_ids


class _FakeActionRobot:
  def __init__(self):
    self.data = SimpleNamespace(
      joint_pos=torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float32),
      default_joint_pos=torch.tensor([[0.1, 0.2], [0.1, 0.2]], dtype=torch.float32),
      encoder_bias=torch.tensor([[0.05, -0.10], [0.0, 0.02]], dtype=torch.float32),
    )
    self.position_target = None
    self.position_joint_ids = None

  def find_joints_by_actuator_names(self, actuator_names):
    assert tuple(actuator_names) == (".*",)
    return [0, 1], ["joint_a", "joint_b"]

  def set_joint_position_target(self, target, joint_ids=None):
    self.position_target = target.clone()
    self.position_joint_ids = joint_ids.clone() if hasattr(joint_ids, "clone") else joint_ids


def _asset_cfg() -> SceneEntityCfg:
  cfg = SceneEntityCfg("robot", body_names=("torso_link",))
  cfg.body_ids = [0]
  return cfg


@pytest.mark.parametrize("terrain", GETUP_TERRAIN_VARIANTS)
def test_host_terrain_variants_are_instantiable(terrain: str) -> None:
  cfg = unitree_g1_getup_env_cfg(terrain=terrain)
  parity = HOST_TERRAIN_PARITY[terrain]
  generator = cfg.scene.terrain.terrain_generator

  assert getattr(cfg, "getup_terrain") == terrain
  assert getattr(cfg, "host_source_task") == f"g1_{terrain}"
  assert getattr(cfg, "host_parity") == parity
  assert generator.num_rows == parity["num_rows"]
  assert generator.num_cols == parity["num_cols"]
  assert generator.curriculum == parity["curriculum"]
  assert cfg.scene.terrain.max_init_terrain_level == parity["max_init_terrain_level"]
  assert cfg.events["getup_assist_force"].params["force_n"] == parity["pull_force_n"]


def test_getup_training_uses_host_scale_parallel_env_count() -> None:
  cfg = unitree_g1_getup_env_cfg(terrain="platform")

  assert cfg.scene.num_envs == 4096


def test_getup_play_keeps_single_env_default_for_interactive_play() -> None:
  cfg = unitree_g1_getup_env_cfg(terrain="platform", play=True)

  assert cfg.scene.num_envs == 1


def test_platform_variant_uses_platform_only_terrain_distribution() -> None:
  cfg = unitree_g1_getup_env_cfg(terrain="platform")
  generator = cfg.scene.terrain.terrain_generator
  proportions = {
    name: sub_terrain.proportion
    for name, sub_terrain in generator.sub_terrains.items()
  }

  assert proportions == {"pyramid_stairs": 1.0}


def test_getup_cfg_uses_host_23dof_relative_action_contract() -> None:
  cfg = unitree_g1_getup_env_cfg(terrain="ground")

  action_cfg = cfg.actions["joint_pos"]
  assert isinstance(action_cfg, HostRelativeJointPositionActionCfg)
  assert action_cfg.scale == 1.0
  assert cfg.terminations.get("head_contact") is None

  robot = Entity(cfg.scene.entities["robot"])
  assert robot.num_joints == 23


def test_host_relative_joint_position_action_targets_current_joint_position() -> None:
  robot = _FakeActionRobot()
  env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
  cfg = HostRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=0.5,
  )
  action = cfg.build(env)

  raw_actions = torch.tensor([[1.0, -1.0], [0.2, 0.4]], dtype=torch.float32)
  action.process_actions(raw_actions)
  action.apply_actions()

  expected = robot.data.joint_pos + raw_actions * 0.5 - robot.data.encoder_bias
  assert torch.allclose(robot.position_target, expected)
  assert not torch.allclose(
    robot.position_target,
    robot.data.default_joint_pos + raw_actions * 0.5 - robot.data.encoder_bias,
  )


def test_getup_height_rewards_use_env_origin_relative_height() -> None:
  asset_cfg = _asset_cfg()
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene=_Scene(
      {"robot": _FakeRobot(body_heights=[0.25, 1.25])},
      env_origins=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    ),
  )

  height_reward = getup_height_progress_reward(
    env,
    min_height=0.12,
    target_height=0.55,
    asset_cfg=asset_cfg,
  )
  posture_reward = getup_posture_reward(
    env,
    torso_height_target=0.25,
    torso_height_std=0.1,
    asset_cfg=asset_cfg,
  )

  assert torch.allclose(height_reward[0], height_reward[1])
  assert torch.allclose(posture_reward[0], posture_reward[1])


def test_getup_assist_force_uses_env_origin_relative_height_gate() -> None:
  asset_cfg = _asset_cfg()
  robot = _FakeRobot(body_heights=[0.25, 1.25])
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene=_Scene(
      {"robot": robot},
      env_origins=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
    ),
  )
  term = apply_getup_assist_force(SimpleNamespace(params={"asset_cfg": asset_cfg}), env)

  term(
    env,
    env_ids=None,
    force_n=100.0,
    activation_height=0.35,
    alignment_threshold=0.0,
    asset_cfg=asset_cfg,
  )

  assert torch.allclose(robot.last_forces[:, 0, 2], torch.tensor([100.0, 100.0]))


def test_stalled_getup_progress_requires_height_and_upright_progress() -> None:
  asset_cfg = _asset_cfg()
  robot = _FakeRobot(body_heights=[0.12, 0.55])
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    step_dt=0.02,
    episode_length_buf=torch.zeros(2, dtype=torch.long),
    scene=_Scene({"robot": robot}, env_origins=torch.zeros(2, 3)),
  )
  term = stalled_getup_progress(None, env)

  terminated = None
  for _ in range(50):
    terminated = term(
      env,
      min_steps_before_check=50,
      progress_threshold=0.2,
      min_height=0.12,
      target_height=0.55,
      asset_cfg=asset_cfg,
    )

  assert terminated.tolist() == [True, False]


def test_prone_variant_is_not_exposed() -> None:
  assert "ground_prone" not in GETUP_TERRAIN_VARIANTS
  with pytest.raises(ValueError):
    unitree_g1_getup_env_cfg(terrain="ground_prone")
