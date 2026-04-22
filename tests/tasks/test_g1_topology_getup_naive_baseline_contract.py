from __future__ import annotations

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner


def test_naive_depth_task_is_registered_and_reuses_isolated_env_family() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0-NaiveDepth")
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0-NaiveDepth")
  assert load_runner_cls("Unitree-G1-TopologyGetUp-Stage0-NaiveDepth") is TopologyGetupOnPolicyRunner
  assert "camera" in cfg.observations
  assert rl_cfg.experiment_name == "g1_topology_getup_naive"


def test_naive_depth_task_uses_plain_spatial_softmax_cnn_model() -> None:
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0-NaiveDepth")
  assert rl_cfg.actor.class_name == "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  assert rl_cfg.critic.class_name == "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  assert "bottleneck_dim" not in rl_cfg.actor.cnn_cfg
