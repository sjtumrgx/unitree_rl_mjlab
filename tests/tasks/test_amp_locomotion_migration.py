"""Migration contract for replacing GetUp/AntiFall-GetUp with AMP locomotion."""

from pathlib import Path

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.amp_loco.rl.runner import AMPOnPolicyRunner


AMP_TASK_IDS = ("Unitree-G1-AMP-Flat", "Unitree-G1-AMP-Rough")
REMOVED_TASK_IDS = (
  "Unitree-G1-GetUp",
  "Unitree-G1-GetUp-AMP",
  "Unitree-G1-GetUp-AMP-LeggedLab",
  "Unitree-G1-AntiFall-GetUp-RecoveryWarmup",
  "Unitree-G1-AntiFall-GetUp",
)


def test_amp_locomotion_tasks_replace_getup_hybrid_tasks() -> None:
  task_ids = set(list_tasks())

  assert set(AMP_TASK_IDS).issubset(task_ids)
  assert set(REMOVED_TASK_IDS).isdisjoint(task_ids)


def test_amp_locomotion_configs_load_motion_groups() -> None:
  for task_id in AMP_TASK_IDS:
    env_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)
    rl_cfg = load_rl_cfg(task_id)

    assert "robot" in env_cfg.scene.entities
    assert "robot" in play_cfg.scene.entities
    assert rl_cfg.experiment_name == "g1_amp_locomotion"
    assert load_runner_cls(task_id) is AMPOnPolicyRunner
    assert env_cfg.amp_motion_files.walk_run
    assert env_cfg.amp_motion_files.recovery
    assert all(Path(path).exists() for path in env_cfg.amp_motion_files.walk_run)
    assert all(Path(path).exists() for path in env_cfg.amp_motion_files.recovery)
