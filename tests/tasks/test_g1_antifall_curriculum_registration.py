import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl.antifall_curriculum import (
  ANTI_FALL_STAGE_TASK_IDS,
  CURRICULUM_TASK_ID,
)
from src.tasks.velocity.rl.curriculum_runner import AntiFallCurriculumRunner
from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner


def test_curriculum_task_is_registered_without_replacing_manual_stages() -> None:
  tasks = set(list_tasks())
  assert CURRICULUM_TASK_ID in tasks
  assert set(ANTI_FALL_STAGE_TASK_IDS).issubset(tasks)


def test_curriculum_task_cfg_defaults_match_contract() -> None:
  env_cfg = load_env_cfg(CURRICULUM_TASK_ID)
  rl_cfg = load_rl_cfg(CURRICULUM_TASK_ID)

  assert load_runner_cls(CURRICULUM_TASK_ID) is AntiFallCurriculumRunner
  assert rl_cfg.max_iterations == 10000
  assert rl_cfg.run_name == "curriculum"
  assert rl_cfg.curriculum.stage_task_ids == ANTI_FALL_STAGE_TASK_IDS
  assert rl_cfg.curriculum.per_stage_max_iterations == 10000
  assert env_cfg.scene.terrain is not None
  assert env_cfg.scene.terrain.terrain_type == "plane"
  assert tuple(env_cfg.observations["actor"].terms) == (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
  )


def test_manual_antifall_stage_tasks_keep_their_original_runner() -> None:
  for task_id in ANTI_FALL_STAGE_TASK_IDS:
    assert load_runner_cls(task_id) is VelocityOnPolicyRunner
