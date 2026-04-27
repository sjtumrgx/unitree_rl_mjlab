import torch


def test_getup_variants_enable_observation_sanitization_and_unstable_guard():
  from mjlab.managers.termination_manager import TerminationTermCfg
  from src.tasks.velocity import mdp
  from src.tasks.velocity.config.g1_getup.env_cfgs import (
    GETUP_TERRAIN_VARIANTS,
    unitree_g1_getup_env_cfg,
  )

  for terrain in GETUP_TERRAIN_VARIANTS:
    cfg = unitree_g1_getup_env_cfg(terrain=terrain)

    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"

    unstable = cfg.terminations.get("unstable_state")
    assert isinstance(unstable, TerminationTermCfg)
    assert unstable.func is mdp.unstable_getup_state

    assert cfg.rewards["body_ang_vel"].func is mdp.bounded_body_angular_velocity_penalty
    assert cfg.rewards["angular_momentum"].func is mdp.bounded_angular_momentum_penalty
    assert cfg.rewards["joint_acc_l2"].func is mdp.bounded_joint_acc_l2
    assert cfg.rewards["action_rate_l2"].func is mdp.bounded_action_rate_after_lift


def test_getup_ppo_uses_tight_action_and_entropy_bounds():
  from src.tasks.velocity.config.g1_getup.rl_cfg import unitree_g1_getup_ppo_runner_cfg

  cfg = unitree_g1_getup_ppo_runner_cfg("ground")

  assert cfg.clip_actions is not None
  assert cfg.clip_actions <= 5.0
  assert cfg.actor.distribution_cfg["init_std"] <= 0.6
  assert cfg.algorithm.entropy_coef <= 0.002


def test_policy_action_sanitizer_clamps_nan_and_inf():
  from src.tasks.velocity.rl.safety import sanitize_policy_actions

  actions = torch.tensor([[float("nan"), float("inf"), -float("inf"), 3.0, -6.0]])
  sanitized = sanitize_policy_actions(actions, clip_actions=5.0)

  assert torch.isfinite(sanitized).all()
  assert sanitized.tolist() == [[0.0, 5.0, -5.0, 3.0, -5.0]]
