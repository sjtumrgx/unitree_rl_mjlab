from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner


def test_teacher_task_is_registered_and_reuses_topology_getup_env_family() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0-Teacher")
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0-Teacher")
  assert load_runner_cls("Unitree-G1-TopologyGetUp-Stage0-Teacher") is TopologyGetupOnPolicyRunner
  assert "camera" in cfg.observations
  assert rl_cfg.experiment_name == "g1_topology_getup_teacher"


def test_teacher_task_uses_richer_critic_plus_camera_observation_sets() -> None:
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0-Teacher")
  assert rl_cfg.obs_groups == {
    "actor": ("critic", "camera"),
    "critic": ("critic", "camera"),
  }
  assert rl_cfg.actor.class_name == "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  assert rl_cfg.critic.class_name == "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
