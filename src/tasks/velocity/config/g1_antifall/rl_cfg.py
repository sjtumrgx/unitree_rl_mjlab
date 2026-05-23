"""RL configuration for Unitree G1 anti-fall velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from src.tasks.velocity.config.g1_getup.rl_cfg import unitree_g1_getup_ppo_runner_cfg

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
  # AntiFall-GetUp resumes two already-useful priors: Stage4b walking and the
  # fallen-start recovery warmup.  The final mixed task exists to teach the
  # switch/resume boundary, not to relearn either prior from scratch.  Keep PPO
  # updates at the same conservative scale as the recovery warmup; local
  # forced-fall probes showed 1e-4 / 0.003 erodes both branches within a few
  # saved updates even when the branch normalizers are gate-separated.
  cfg.algorithm.learning_rate = 1.0e-5
  cfg.algorithm.desired_kl = 0.001
  # Keep the same raw-action envelope as BFM-Zero/GetUp.  Without this explicit
  # clip the G1 walking base config leaves actions unclipped, so a rescaled
  # recovery prior can emit very large raw values before the 0.25 physical
  # recovery scale and env-side delta clamps see them.
  cfg.clip_actions = 5.0
  cfg.actor.class_name = "src.tasks.velocity.rl.gated_actor:GatedAntiFallGetUpActor"
  cfg.actor.distribution_cfg["init_std"] = 0.5
  return cfg


def unitree_g1_antifall_getup_recovery_warmup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  # This branch is a fallen-start GetUp bootstrap using the final AntiFall-GetUp
  # actor/action tensor contract.  It is normally actor-only resumed from the
  # proven standalone GetUp policy, so keep the physical action clip but make
  # the first PPO updates much smaller than the from-scratch GetUp bootstrap.
  # Local probes showed the 1e-3 GetUp LR can erase the recovery prior within
  # the first saved update, while a 1e-5 / 0.001 KL update preserves rollout
  # get-up behavior and still lets the 29-DoF recovery branch adapt.
  cfg = unitree_g1_getup_ppo_runner_cfg(terrain="mixed")
  cfg.experiment_name = "g1_antifall_getup"
  cfg.run_name = "recovery_warmup"
  cfg.algorithm.learning_rate = 1.0e-5
  cfg.algorithm.desired_kl = 0.001
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
