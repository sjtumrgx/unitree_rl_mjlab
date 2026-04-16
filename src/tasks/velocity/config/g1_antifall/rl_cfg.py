"""RL configuration for Unitree G1 anti-fall velocity tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def unitree_g1_antifall_ppo_runner_cfg(stage_name: str = "") -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_antifall"
  cfg.run_name = stage_name
  return cfg
