"""RL cfg placeholder for the dedicated G1 parkour ONNX play task."""

from __future__ import annotations

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_parkour_runner_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_parkour_flat_debug"
  cfg.run_name = "onnx_play_contract"
  return cfg
