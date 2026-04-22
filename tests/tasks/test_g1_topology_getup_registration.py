from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from src.tasks.velocity.rl.topology_getup_runner import TopologyGetupOnPolicyRunner

_EXPECTED_TASK_IDS = (
  "Unitree-G1-TopologyGetUp-Stage0",
  "Unitree-G1-TopologyGetUp-Benchmark",
)


def test_g1_topology_getup_tasks_are_registered_without_replacing_existing_antifall_tasks() -> None:
  task_ids = list_tasks()
  for task_id in _EXPECTED_TASK_IDS:
    assert task_id in task_ids
  assert "Unitree-G1-AntiFall-Stage0" in task_ids
  assert "Unitree-G1-AntiFall-Benchmark" in task_ids


def test_g1_topology_getup_task_cfgs_load_for_train_and_play() -> None:
  for task_id in _EXPECTED_TASK_IDS:
    train_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)
    rl_cfg = load_rl_cfg(task_id)
    assert train_cfg.scene.terrain is not None
    assert play_cfg.scene.terrain is not None
    assert "camera" in train_cfg.observations
    assert "camera" in play_cfg.observations
    assert rl_cfg.experiment_name == "g1_topology_getup"
    assert load_runner_cls(task_id) is TopologyGetupOnPolicyRunner
