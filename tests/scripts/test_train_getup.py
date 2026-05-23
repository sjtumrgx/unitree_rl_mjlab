from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg


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
    resume=True,
    load_run="2026-05-18_14-27-39_ground",
    load_checkpoint="model_9999.pt",
    clip_actions=3.0,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    actor=object(),
    critic=object(),
    algorithm=object(),
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
    resume=False,
    load_run=".*",
    load_checkpoint="model_.*.pt",
    clip_actions=5.0,
    obs_groups={},
    actor=object(),
    critic=object(),
    algorithm=object(),
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
  assert launched.agent.resume is True
  assert launched.agent.load_run == "2026-05-18_14-27-39_ground"
  assert launched.agent.load_checkpoint == "model_9999.pt"
  assert launched.agent.clip_actions == 3.0
  assert launched.agent.obs_groups is base_agent.obs_groups
  assert launched.agent.actor is base_agent.actor
  assert launched.agent.critic is base_agent.critic
  assert launched.agent.algorithm is base_agent.algorithm


def test_getup_terrain_launch_preserves_nested_ppo_overrides(monkeypatch) -> None:
  from scripts import train

  captured = {}

  monkeypatch.setattr(train, "select_gpus", lambda gpu_ids: (None, 0))
  monkeypatch.setattr(
    train,
    "run_train",
    lambda task_id, args, log_dir: captured.update(task_id=task_id, args=args, log_dir=log_dir),
  )

  args = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    gpu_ids="cpu",
    getup_terrain="platform",
  )
  args.agent.algorithm.learning_rate = 1.0e-4
  args.agent.algorithm.desired_kl = 0.003
  args.agent.actor.distribution_cfg["init_std"] = 0.2

  train.launch_training("Unitree-G1-GetUp", args)

  launched = captured["args"]
  assert launched.agent.algorithm.learning_rate == pytest.approx(1.0e-4)
  assert launched.agent.algorithm.desired_kl == pytest.approx(0.003)
  assert launched.agent.actor.distribution_cfg["init_std"] == pytest.approx(0.2)


def test_run_train_actor_only_resume_loads_only_actor(monkeypatch, tmp_path) -> None:
  from scripts import train

  captured = {}

  class _FakeEnv:
    def close(self):
      captured["closed"] = True

  class _FakeRunner:
    def __init__(self, env, agent_cfg, log_dir, device, **kwargs):
      captured["runner_init"] = {
        "env": env,
        "agent_cfg": agent_cfg,
        "log_dir": log_dir,
        "device": device,
        "kwargs": kwargs,
      }

    def add_git_repo_to_log(self, path):
      captured["git_log_path"] = path

    def load(self, path, load_cfg=None):
      captured["load"] = {"path": path, "load_cfg": load_cfg}

    def learn(self, num_learning_iterations, init_at_random_ep_len):
      captured["learn"] = {
        "num_learning_iterations": num_learning_iterations,
        "init_at_random_ep_len": init_at_random_ep_len,
      }

  agent = RslRlOnPolicyRunnerCfg(
    seed=3,
    resume=True,
    load_run="old_run",
    load_checkpoint="model_100.pt",
    clip_actions=5.0,
    max_iterations=7,
  )
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    actor_only_resume=True,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "get_checkpoint_path", lambda root, run, ckpt: tmp_path / run / ckpt)
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)

  train.run_train("Unitree-G1-GetUp", cfg, tmp_path / "logs" / "rsl_rl" / "g1_getup" / "new_run")

  assert captured["load"] == {
    "path": str(tmp_path / "old_run" / "model_100.pt"),
    "load_cfg": {"actor": True},
  }
  assert captured["learn"] == {"num_learning_iterations": 7, "init_at_random_ep_len": True}
  assert captured["closed"] is True


def test_run_train_can_resume_from_explicit_checkpoint_path(monkeypatch, tmp_path) -> None:
  from scripts import train

  captured = {}

  class _FakeEnv:
    def close(self):
      captured["closed"] = True

  class _FakeRunner:
    def __init__(self, *args, **kwargs):
      del args, kwargs

    def add_git_repo_to_log(self, path):
      del path

    def load(self, path, load_cfg=None):
      captured["load"] = {"path": path, "load_cfg": load_cfg}

    def learn(self, num_learning_iterations, init_at_random_ep_len):
      captured["learn"] = (num_learning_iterations, init_at_random_ep_len)

  checkpoint = tmp_path / "logs" / "rsl_rl" / "g1_getup" / "good" / "model_1000.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.write_bytes(b"checkpoint")
  agent = RslRlOnPolicyRunnerCfg(
    resume=True,
    load_run="should-not-be-used",
    load_checkpoint="should-not-be-used.pt",
    max_iterations=1,
  )
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    resume_checkpoint_path=str(checkpoint),
    actor_only_resume=True,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "get_checkpoint_path", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected regex checkpoint lookup")))
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)

  train.run_train("Unitree-G1-GetUp", cfg, tmp_path / "logs" / "rsl_rl" / "g1_getup" / "new_run")

  assert captured["load"] == {
    "path": str(checkpoint),
    "load_cfg": {"actor": True},
  }
  assert captured["closed"] is True



def test_run_train_policy_only_resume_loads_actor_and_critic_without_optimizer(monkeypatch, tmp_path) -> None:
  from scripts import train

  captured = {}

  class _FakeEnv:
    def close(self):
      captured["closed"] = True

  class _FakeRunner:
    def __init__(self, *args, **kwargs):
      del args, kwargs

    def add_git_repo_to_log(self, path):
      del path

    def load(self, path, load_cfg=None):
      captured["load"] = {"path": path, "load_cfg": load_cfg}

    def learn(self, num_learning_iterations, init_at_random_ep_len):
      captured["learn"] = (num_learning_iterations, init_at_random_ep_len)

  checkpoint = tmp_path / "logs" / "rsl_rl" / "g1_getup" / "good" / "model_1000.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.write_bytes(b"checkpoint")
  agent = RslRlOnPolicyRunnerCfg(
    resume=True,
    load_run="should-not-be-used",
    load_checkpoint="should-not-be-used.pt",
    max_iterations=1,
  )
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    resume_checkpoint_path=str(checkpoint),
    policy_only_resume=True,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "get_checkpoint_path", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected regex checkpoint lookup")))
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)

  train.run_train("Unitree-G1-GetUp", cfg, tmp_path / "logs" / "rsl_rl" / "g1_getup" / "new_run")

  assert captured["load"] == {
    "path": str(checkpoint),
    "load_cfg": {"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
  }
  assert captured["closed"] is True

def test_run_train_can_reset_actor_std_after_actor_only_resume(monkeypatch, tmp_path) -> None:
  from scripts import train

  captured = {}

  class _FakeEnv:
    def close(self):
      pass

  class _FakeDistribution:
    def __init__(self):
      self.std_param = torch.zeros(3)

  class _FakePolicy:
    def __init__(self):
      self.distribution = _FakeDistribution()

  class _FakeAlg:
    def __init__(self):
      self.policy = _FakePolicy()

    def get_policy(self):
      return self.policy

  class _FakeRunner:
    def __init__(self, *args, **kwargs):
      del args, kwargs
      self.alg = _FakeAlg()
      captured["runner"] = self

    def add_git_repo_to_log(self, path):
      del path

    def load(self, path, load_cfg=None):
      captured["load"] = {"path": path, "load_cfg": load_cfg}

    def learn(self, num_learning_iterations, init_at_random_ep_len):
      captured["learn"] = (num_learning_iterations, init_at_random_ep_len)

  agent = RslRlOnPolicyRunnerCfg(
    resume=True,
    load_run="old_run",
    load_checkpoint="model_100.pt",
    max_iterations=1,
  )
  agent.actor.distribution_cfg["init_std"] = 0.5
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    actor_only_resume=True,
    reset_actor_std_on_resume=True,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "get_checkpoint_path", lambda root, run, ckpt: tmp_path / run / ckpt)
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)

  train.run_train("Unitree-G1-GetUp", cfg, tmp_path / "logs" / "rsl_rl" / "g1_getup" / "new_run")

  std_param = captured["runner"].alg.get_policy().distribution.std_param
  assert torch.allclose(std_param, torch.full((3,), 0.5))
  assert captured["load"]["load_cfg"] == {"actor": True}


def test_run_train_rejects_policy_and_actor_only_resume_together(monkeypatch, tmp_path) -> None:
  from scripts import train

  class _FakeEnv:
    def close(self):
      pass

  class _FakeRunner:
    def __init__(self, *args, **kwargs):
      del args, kwargs

    def add_git_repo_to_log(self, path):
      del path

    def load(self, path, load_cfg=None):  # pragma: no cover - should fail before load
      raise AssertionError("resume flags should be rejected before loading")

  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  agent = RslRlOnPolicyRunnerCfg(resume=True, max_iterations=1)
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    resume_checkpoint_path=str(checkpoint),
    actor_only_resume=True,
    policy_only_resume=True,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)

  with pytest.raises(ValueError, match="mutually exclusive"):
    train.run_train("Unitree-G1-GetUp", cfg, tmp_path / "logs")


def test_expand_actor_checkpoint_input_preserves_old_policy_and_zeroes_new_bfm_inputs() -> None:
  from scripts.train import _expand_model_input_state

  checkpoint_state = {
    "obs_normalizer._mean": torch.full((1, 492), 0.1),
    "obs_normalizer._var": torch.full((1, 492), 2.0),
    "obs_normalizer._std": torch.full((1, 492), 3.0),
    "mlp.0.weight": torch.ones(512, 492),
    "mlp.0.bias": torch.arange(512, dtype=torch.float32),
    "mlp.2.weight": torch.ones(256, 512),
  }
  target_state = {
    "obs_normalizer._mean": torch.zeros(1, 850),
    "obs_normalizer._var": torch.ones(1, 850),
    "obs_normalizer._std": torch.ones(1, 850),
    "mlp.0.weight": torch.full((512, 850), -7.0),
    "mlp.0.bias": torch.full((512,), -3.0),
    "mlp.2.weight": torch.full((256, 512), 4.0),
  }

  expanded = _expand_model_input_state(checkpoint_state, target_state)

  assert expanded is True
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, :492], torch.ones(512, 492))
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 492:], torch.zeros(512, 358))
  torch.testing.assert_close(checkpoint_state["mlp.0.bias"], torch.arange(512, dtype=torch.float32))
  torch.testing.assert_close(checkpoint_state["mlp.2.weight"], torch.ones(256, 512))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, :492], torch.full((1, 492), 0.1))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, 492:], torch.zeros(1, 358))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._var"][:, :492], torch.full((1, 492), 2.0))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._var"][:, 492:], torch.ones(1, 358))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._std"][:, :492], torch.full((1, 492), 3.0))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._std"][:, 492:], torch.ones(1, 358))


def test_expand_getup_actor_output_maps_23dof_common_joints_into_29dof_policy() -> None:
  from scripts.train import _expand_model_input_state

  source_joints = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
  )
  target_joints = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  )
  old_head = torch.arange(23 * 4, dtype=torch.float32).reshape(23, 4)
  old_bias = torch.arange(23, dtype=torch.float32)
  checkpoint_state = {
    "distribution.std_param": torch.full((23,), 0.25),
    "mlp.6.weight": old_head.clone(),
    "mlp.6.bias": old_bias.clone(),
  }
  target_state = {
    "distribution.std_param": torch.full((29,), 0.5),
    "mlp.6.weight": torch.full((29, 4), -9.0),
    "mlp.6.bias": torch.full((29,), -3.0),
  }

  expanded = _expand_model_input_state(
    checkpoint_state,
    target_state,
    source_action_names=source_joints,
    target_action_names=target_joints,
  )

  assert expanded is True
  source_by_name = {name: idx for idx, name in enumerate(source_joints)}
  for target_idx, name in enumerate(target_joints):
    if name in source_by_name:
      source_idx = source_by_name[name]
      torch.testing.assert_close(checkpoint_state["mlp.6.weight"][target_idx], old_head[source_idx])
      assert checkpoint_state["mlp.6.bias"][target_idx] == old_bias[source_idx]
      assert checkpoint_state["distribution.std_param"][target_idx] == pytest.approx(0.25)
    else:
      torch.testing.assert_close(checkpoint_state["mlp.6.weight"][target_idx], torch.zeros(4))
      assert checkpoint_state["mlp.6.bias"][target_idx] == pytest.approx(0.0)
      assert checkpoint_state["distribution.std_param"][target_idx] == pytest.approx(0.5)


def test_expand_getup_actor_output_infers_g1_23_to_29dof_joint_names() -> None:
  from scripts.train import _expand_model_input_state

  old_head = torch.arange(23 * 4, dtype=torch.float32).reshape(23, 4)
  checkpoint_state = {
    "distribution.std_param": torch.full((23,), 0.25),
    "mlp.6.weight": old_head.clone(),
    "mlp.6.bias": torch.arange(23, dtype=torch.float32),
  }
  target_state = {
    "distribution.std_param": torch.full((29,), 0.5),
    "mlp.6.weight": torch.full((29, 4), -9.0),
    "mlp.6.bias": torch.full((29,), -3.0),
  }

  expanded = _expand_model_input_state(checkpoint_state, target_state)

  assert expanded is True
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][0], old_head[0])
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][12], old_head[12])
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][13], torch.zeros(4))
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][14], torch.zeros(4))
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][15], old_head[13])
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][22], old_head[18])
  assert checkpoint_state["distribution.std_param"][13] == pytest.approx(0.5)
  assert checkpoint_state["mlp.6.bias"][13] == pytest.approx(0.0)
