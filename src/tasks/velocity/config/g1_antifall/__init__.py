from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl.antifall_runner import AntiFallOnPolicyRunner
from src.tasks.velocity.rl.curriculum_runner import AntiFallCurriculumRunner

from src.tasks.velocity.rl.antifall_curriculum import CURRICULUM_TASK_ID
from .env_cfgs import (
  unitree_g1_antifall_benchmark_env_cfg,
  unitree_g1_antifall_stage0_env_cfg,
  unitree_g1_antifall_stage1_env_cfg,
  unitree_g1_antifall_stage2_env_cfg,
  unitree_g1_antifall_stage3_env_cfg,
  unitree_g1_antifall_stage4a_env_cfg,
  unitree_g1_antifall_stage4b_env_cfg,
)
from .rl_cfg import (
  unitree_g1_antifall_curriculum_runner_cfg,
  unitree_g1_antifall_ppo_runner_cfg,
)


def _register_antifall_task(task_id: str, stage_name: str, env_cfg_factory) -> None:
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg_factory(),
    play_env_cfg=env_cfg_factory(play=True),
    rl_cfg=unitree_g1_antifall_ppo_runner_cfg(stage_name=stage_name),
    runner_cls=AntiFallOnPolicyRunner,
  )


_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Stage0",
  stage_name="stage0",
  env_cfg_factory=unitree_g1_antifall_stage0_env_cfg,
)
_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Stage1",
  stage_name="stage1",
  env_cfg_factory=unitree_g1_antifall_stage1_env_cfg,
)
_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Stage2",
  stage_name="stage2",
  env_cfg_factory=unitree_g1_antifall_stage2_env_cfg,
)
_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Stage3",
  stage_name="stage3",
  env_cfg_factory=unitree_g1_antifall_stage3_env_cfg,
)
_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Stage4a",
  stage_name="stage4a",
  env_cfg_factory=unitree_g1_antifall_stage4a_env_cfg,
)
_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Stage4b",
  stage_name="stage4b",
  env_cfg_factory=unitree_g1_antifall_stage4b_env_cfg,
)
_register_antifall_task(
  task_id="Unitree-G1-AntiFall-Benchmark",
  stage_name="benchmark",
  env_cfg_factory=unitree_g1_antifall_benchmark_env_cfg,
)

register_mjlab_task(
  task_id=CURRICULUM_TASK_ID,
  env_cfg=unitree_g1_antifall_stage0_env_cfg(),
  play_env_cfg=unitree_g1_antifall_stage0_env_cfg(play=True),
  rl_cfg=unitree_g1_antifall_curriculum_runner_cfg(),
  runner_cls=AntiFallCurriculumRunner,
)
