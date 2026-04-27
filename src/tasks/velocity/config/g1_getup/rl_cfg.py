"""RL configuration for Unitree G1 HoST get-up tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg



def unitree_g1_getup_ppo_runner_cfg(terrain: str = "ground") -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner configuration for Unitree G1 HoST get-up."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_getup",
    run_name=terrain,
    # HoST clips env actions before storing them in observations and before
    # applying relative joint targets.  Without this bound, learned Gaussian
    # means can explode, feeding huge raw actions into `last_action` and
    # `action_rate_l2` until the actor observation becomes non-finite.
    clip_actions=100.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
