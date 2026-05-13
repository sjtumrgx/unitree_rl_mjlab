from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace


def test_getup_terrain_launch_preserves_cli_resource_overrides(monkeypatch) -> None:
  from scripts import train

  base_env = SimpleNamespace(scene=SimpleNamespace(num_envs=4), seed=7)
  base_agent = SimpleNamespace(
    max_iterations=1,
    num_steps_per_env=2,
    save_interval=123,
    logger="tensorboard",
    upload_model=False,
    seed=7,
    experiment_name="g1_getup",
    run_name="ground",
  )
  terrain_env = SimpleNamespace(scene=SimpleNamespace(num_envs=4096), seed=0)
  terrain_agent = SimpleNamespace(
    max_iterations=10001,
    num_steps_per_env=24,
    save_interval=100,
    logger="wandb",
    upload_model=True,
    seed=0,
    experiment_name="g1_getup",
    run_name="ground",
  )
  captured = {}

  monkeypatch.setattr(
    train,
    "select_gpus",
    lambda gpu_ids: (None, 0),
  )
  monkeypatch.setattr(
    train,
    "run_train",
    lambda task_id, args, log_dir: captured.update(task_id=task_id, args=args, log_dir=log_dir),
  )

  import src.tasks.velocity.config.g1_getup.env_cfgs as env_cfgs
  import src.tasks.velocity.config.g1_getup.rl_cfg as rl_cfg

  monkeypatch.setattr(env_cfgs, "unitree_g1_getup_env_cfg", lambda terrain: terrain_env)
  monkeypatch.setattr(rl_cfg, "unitree_g1_getup_ppo_runner_cfg", lambda terrain: terrain_agent)

  args = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    env=base_env,
    agent=base_agent,
    gpu_ids="cpu",
    getup_terrain="ground",
  )

  train.launch_training("Unitree-G1-GetUp", args)

  launched = captured["args"]
  assert captured["task_id"] == "Unitree-G1-GetUp"
  assert launched.env.scene.num_envs == 4
  assert launched.agent.max_iterations == 1
  assert launched.agent.num_steps_per_env == 2
  assert launched.agent.save_interval == 123
  assert launched.agent.logger == "tensorboard"
  assert launched.agent.upload_model is False
  assert launched.agent.seed == 7
