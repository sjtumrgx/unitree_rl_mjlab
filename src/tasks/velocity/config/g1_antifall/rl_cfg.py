"""RL configuration for Unitree G1 anti-fall velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from src.tasks.velocity.rl.antifall_curriculum import AntiFallCurriculumCfg


def unitree_g1_antifall_ppo_runner_cfg(stage_name: str = "") -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_antifall"
  cfg.run_name = stage_name
  return cfg


def unitree_g1_antifall_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_antifall_ppo_runner_cfg(stage_name="antifall_getup")
  cfg.experiment_name = "g1_antifall_getup"
  cfg.max_iterations = 10001
  cfg.save_interval = 100
  # AntiFall-GetUp resumes the stage4b walking actor and only fine-tunes it for
  # fall recovery.  The base velocity PPO defaults are intentionally aggressive
  # for from-scratch walking, but they overwrite the warm-start too quickly here
  # and make the policy lose pre-push tracking before recovery has a chance to
  # improve.  Keep this branch conservative unless an explicit from-scratch
  # GetUp curriculum is added.
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.desired_kl = 0.003
  cfg.actor.distribution_cfg["init_std"] = 0.5
  return cfg


@dataclass
class AntiFallCurriculumRunnerCfg(RslRlOnPolicyRunnerCfg):
  curriculum: AntiFallCurriculumCfg = field(default_factory=AntiFallCurriculumCfg)


def unitree_g1_antifall_curriculum_runner_cfg() -> AntiFallCurriculumRunnerCfg:
  base = unitree_g1_ppo_runner_cfg()
  return AntiFallCurriculumRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=10000,
    obs_groups=base.obs_groups,
    save_interval=base.save_interval,
    experiment_name="g1_antifall_curriculum",
    run_name="curriculum",
    logger=base.logger,
    wandb_project=base.wandb_project,
    wandb_tags=base.wandb_tags,
    resume=base.resume,
    load_run=base.load_run,
    load_checkpoint=base.load_checkpoint,
    clip_actions=base.clip_actions,
    upload_model=base.upload_model,
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
  )
