"""RL configuration for Unitree G1 HoST get-up tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class GetupAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Serializable AMP extension fields for the opt-in GetUp fallback."""

  class_name: str = "src.tasks.velocity.rl.getup_amp:GetupAmpPPO"
  demo_data_dir: str = "data/motions/g1_getup_amp"
  manifest_path: str | None = None
  amp_reward_scale: float = 0.25
  amp_obs_group: str = "amp"
  discriminator_hidden_dims: tuple[int, ...] = (256, 128)
  discriminator_learning_rate: float = 1.0e-4
  amp_batch_size: int = 256
  amp_buffer_capacity: int = 65536
  discriminator_grad_penalty: float = 0.0
  require_demo_data: bool = True
  # Demo resampling target step.  Overridden at construct time with the env's
  # actual step_dt so the discriminator never sees mismatched temporal scales.
  amp_target_dt: float = 0.02
  amp_getup_segments: bool = True
  amp_feature_layout: str = "yaw_invariant"


def unitree_g1_getup_ppo_runner_cfg(terrain: str = "mixed") -> RslRlOnPolicyRunnerCfg:
  """Create PPO runner configuration for Unitree G1 HoST get-up."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
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
      entropy_coef=0.001,
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
    # Get-up actions are deltas from the current joint pose.  HoST exposed a
    # broad outer clip, but MJLab PPO entropy can raise std enough that rare
    # samples request multi-radian pose jumps and inject contact explosions.
    # Keep exploration inside a physically recoverable range before action
    # history, action-rate rewards, and target writes see the sample.
    clip_actions=5.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_g1_getup_amp_ppo_runner_cfg(
  demo_data_dir: str = "data/motions/g1_getup_amp",
  manifest_path: str | None = None,
  amp_reward_scale: float = 0.25,
) -> RslRlOnPolicyRunnerCfg:
  """Create the opt-in ground-only AMP runner configuration."""
  cfg = unitree_g1_getup_ppo_runner_cfg(terrain="ground")
  cfg.algorithm = GetupAmpPpoAlgorithmCfg(
    value_loss_coef=cfg.algorithm.value_loss_coef,
    use_clipped_value_loss=cfg.algorithm.use_clipped_value_loss,
    clip_param=cfg.algorithm.clip_param,
    entropy_coef=cfg.algorithm.entropy_coef,
    num_learning_epochs=cfg.algorithm.num_learning_epochs,
    num_mini_batches=cfg.algorithm.num_mini_batches,
    learning_rate=cfg.algorithm.learning_rate,
    schedule=cfg.algorithm.schedule,
    gamma=cfg.algorithm.gamma,
    lam=cfg.algorithm.lam,
    desired_kl=cfg.algorithm.desired_kl,
    max_grad_norm=cfg.algorithm.max_grad_norm,
    demo_data_dir=demo_data_dir,
    manifest_path=manifest_path,
    amp_reward_scale=amp_reward_scale,
  )
  cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",), "amp": ("amp",)}
  cfg.experiment_name = "g1_getup_amp"
  cfg.run_name = "ground_amp"
  cfg.save_interval = 50
  return cfg
