"""Task registration for Unitree G1 terrain get-up scaffolds.

These registrations are intentionally separate from the existing anti-fall task IDs.
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl.topology_getup_distillation_runner import TopologyGetupDistillationRunner
from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner

from .env_cfgs import (
  unitree_g1_topology_getup_benchmark_env_cfg,
  unitree_g1_topology_getup_stage0_env_cfg,
)
from .rl_cfg import (
  unitree_g1_topology_getup_distillation_runner_cfg,
  unitree_g1_topology_getup_naive_ppo_runner_cfg,
  unitree_g1_topology_getup_ppo_runner_cfg,
  unitree_g1_topology_getup_teacher_ppo_runner_cfg,
)


def _register_topology_getup_task(task_id: str, stage_name: str, env_cfg_factory, *, naive: bool = False, teacher: bool = False) -> None:
  rl_cfg = (
    unitree_g1_topology_getup_teacher_ppo_runner_cfg(stage_name=stage_name)
    if teacher
    else unitree_g1_topology_getup_naive_ppo_runner_cfg(stage_name=stage_name)
    if naive
    else unitree_g1_topology_getup_ppo_runner_cfg(stage_name=stage_name)
  )
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg_factory(),
    play_env_cfg=env_cfg_factory(play=True),
    rl_cfg=rl_cfg,
    runner_cls=TopologyGetupOnPolicyRunner,
  )


_register_topology_getup_task(
  task_id="Unitree-G1-TopologyGetUp-Stage0",
  stage_name="stage0",
  env_cfg_factory=unitree_g1_topology_getup_stage0_env_cfg,
)
_register_topology_getup_task(
  task_id="Unitree-G1-TopologyGetUp-Benchmark",
  stage_name="benchmark",
  env_cfg_factory=unitree_g1_topology_getup_benchmark_env_cfg,
)

_register_topology_getup_task(
  task_id="Unitree-G1-TopologyGetUp-Stage0-NaiveDepth",
  stage_name="stage0_naive_depth",
  env_cfg_factory=unitree_g1_topology_getup_stage0_env_cfg,
  naive=True,
)

_register_topology_getup_task(
  task_id="Unitree-G1-TopologyGetUp-Stage0-Teacher",
  stage_name="stage0_teacher",
  env_cfg_factory=unitree_g1_topology_getup_stage0_env_cfg,
  teacher=True,
)


register_mjlab_task(
  task_id="Unitree-G1-TopologyGetUp-Stage0-Distill",
  env_cfg=unitree_g1_topology_getup_stage0_env_cfg(),
  play_env_cfg=unitree_g1_topology_getup_stage0_env_cfg(play=True),
  rl_cfg=unitree_g1_topology_getup_distillation_runner_cfg(stage_name="stage0_distill"),
  runner_cls=TopologyGetupDistillationRunner,
)
