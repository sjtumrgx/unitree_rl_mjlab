import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl import VelocityOnPolicyRunner


def test_g1_getup_is_registered_and_old_getup_tasks_are_absent() -> None:
  task_ids = list_tasks()
  assert "Unitree-G1-GetUp" in task_ids
  assert not [task_id for task_id in task_ids if "Unitree-G1-" + "Topology" + "GetUp" in task_id]
  assert "Unitree-G1-AntiFall-Stage0" in task_ids
  assert "Unitree-G1-AntiFall-Benchmark" in task_ids


def test_g1_getup_cfgs_load_for_train_and_play() -> None:
  train_cfg = load_env_cfg("Unitree-G1-GetUp")
  play_cfg = load_env_cfg("Unitree-G1-GetUp", play=True)
  rl_cfg = load_rl_cfg("Unitree-G1-GetUp")

  assert "robot" in train_cfg.scene.entities
  assert "robot" in play_cfg.scene.entities
  assert getattr(train_cfg, "getup_terrain") == "ground"
  assert rl_cfg.experiment_name == "g1_getup"
  assert "topology" not in rl_cfg.experiment_name.lower()
  assert load_runner_cls("Unitree-G1-GetUp") is VelocityOnPolicyRunner
