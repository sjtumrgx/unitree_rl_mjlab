import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.rl.antifall_runner import AntiFallOnPolicyRunner

EXPECTED_TASK_IDS = (
  "Unitree-G1-AntiFall-Stage0",
  "Unitree-G1-AntiFall-Stage1",
  "Unitree-G1-AntiFall-Stage2",
  "Unitree-G1-AntiFall-Stage3",
  "Unitree-G1-AntiFall-Stage4a",
  "Unitree-G1-AntiFall-Stage4b",
  "Unitree-G1-AntiFall-Benchmark",
)


def test_g1_antifall_tasks_are_registered() -> None:
  assert set(EXPECTED_TASK_IDS).issubset(set(list_tasks()))


def test_g1_antifall_task_cfgs_load_for_train_and_play() -> None:
  for task_id in EXPECTED_TASK_IDS:
    train_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)
    rl_cfg = load_rl_cfg(task_id)

    assert "robot" in train_cfg.scene.entities
    assert "robot" in play_cfg.scene.entities
    assert rl_cfg.experiment_name == "g1_antifall"
    assert rl_cfg.run_name
    assert load_runner_cls(task_id) is AntiFallOnPolicyRunner
