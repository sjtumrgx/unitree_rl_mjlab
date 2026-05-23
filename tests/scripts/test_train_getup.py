from __future__ import annotations

from dataclasses import replace
import math
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


def test_run_train_rejects_explicit_checkpoint_path_without_resume(monkeypatch, tmp_path) -> None:
  from scripts import train

  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    gpu_ids="cpu",
    resume_checkpoint_path=str(checkpoint),
  )
  assert cfg.agent.resume is False

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(
    train,
    "ManagerBasedRlEnv",
    lambda **kwargs: (_ for _ in ()).throw(AssertionError("env should not be created")),
  )

  with pytest.raises(ValueError, match="resume_checkpoint_path requires agent.resume=True"):
    train.run_train("Unitree-G1-GetUp", cfg, tmp_path / "logs")



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


def test_run_train_fuses_walking_and_recovery_actor_checkpoints(monkeypatch, tmp_path) -> None:
  from scripts import train

  captured = {}

  class _FakeEnv:
    def close(self):
      captured["closed"] = True

  class _FakeDistribution:
    def __init__(self):
      self.std_param = torch.full((3,), 9.0)

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

    def load(self, path, load_cfg=None):  # pragma: no cover - fusion path must not use generic load
      raise AssertionError(f"unexpected generic load: {path=} {load_cfg=}")

    def learn(self, num_learning_iterations, init_at_random_ep_len):
      captured["learn"] = (num_learning_iterations, init_at_random_ep_len)

  walking_checkpoint = tmp_path / "walking.pt"
  recovery_checkpoint = tmp_path / "recovery.pt"
  walking_checkpoint.write_bytes(b"walking")
  recovery_checkpoint.write_bytes(b"recovery")
  agent = RslRlOnPolicyRunnerCfg(resume=True, max_iterations=1)
  agent.actor.distribution_cfg["init_std"] = 0.5
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    resume_checkpoint_path=str(walking_checkpoint),
    recovery_resume_checkpoint_path=str(recovery_checkpoint),
    actor_only_resume=True,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)
  monkeypatch.setattr(
    train,
    "_load_fused_antifall_getup_actor",
    lambda runner, *, walking_resume_path, recovery_resume_path, target_env_cfg, map_location: captured.update(
      fused={
        "runner": runner,
        "walking_resume_path": walking_resume_path,
        "recovery_resume_path": recovery_resume_path,
        "target_env_cfg": target_env_cfg,
        "map_location": map_location,
      }
    ),
  )

  train.run_train("Unitree-G1-AntiFall-GetUp", cfg, tmp_path / "logs")

  assert captured["fused"]["walking_resume_path"] == walking_checkpoint
  assert captured["fused"]["recovery_resume_path"] == recovery_checkpoint
  assert captured["fused"]["target_env_cfg"] is cfg.env
  assert captured["fused"]["map_location"] == "cpu"
  std_param = captured["runner"].alg.get_policy().distribution.std_param
  assert torch.allclose(std_param, torch.full((3,), 0.5))
  assert captured["learn"] == (1, True)
  assert captured["closed"] is True


def test_run_train_rejects_recovery_checkpoint_without_actor_only_resume(monkeypatch, tmp_path) -> None:
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
      raise AssertionError("fusion flags should be rejected before loading")

  walking_checkpoint = tmp_path / "walking.pt"
  recovery_checkpoint = tmp_path / "recovery.pt"
  walking_checkpoint.write_bytes(b"walking")
  recovery_checkpoint.write_bytes(b"recovery")
  agent = RslRlOnPolicyRunnerCfg(resume=True, max_iterations=1)
  cfg = replace(
    train.TrainConfig.from_task("Unitree-G1-GetUp"),
    agent=agent,
    gpu_ids="cpu",
    resume_checkpoint_path=str(walking_checkpoint),
    recovery_resume_checkpoint_path=str(recovery_checkpoint),
    actor_only_resume=False,
  )

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(train, "configure_torch_backends", lambda: None)
  monkeypatch.setattr(train, "ManagerBasedRlEnv", lambda **kwargs: _FakeEnv())
  monkeypatch.setattr(train, "FiniteActionRslRlVecEnvWrapper", lambda env, clip_actions: env)
  monkeypatch.setattr(train, "load_runner_cls", lambda task_id: _FakeRunner)
  monkeypatch.setattr(train, "dump_yaml", lambda *args, **kwargs: None)

  with pytest.raises(ValueError, match="requires actor_only_resume=True"):
    train.run_train("Unitree-G1-AntiFall-GetUp", cfg, tmp_path / "logs")


def test_antifall_getup_ppo_uses_gated_dual_actor() -> None:
  from src.tasks.velocity.config.g1_antifall.rl_cfg import unitree_g1_antifall_getup_ppo_runner_cfg

  cfg = unitree_g1_antifall_getup_ppo_runner_cfg()

  assert cfg.actor.class_name == "src.tasks.velocity.rl.gated_actor:GatedAntiFallGetUpActor"
  assert cfg.algorithm.learning_rate == pytest.approx(1.0e-5)
  assert cfg.algorithm.desired_kl == pytest.approx(0.001)


def test_gated_antifall_getup_actor_selects_recovery_branch_from_latest_getup_progress() -> None:
  from tensordict import TensorDict

  from scripts import train
  from src.tasks.velocity.rl.gated_actor import GatedAntiFallGetUpActor

  obs_width = train._layout_width(train._g1_antifall_getup_actor_layout())
  obs = TensorDict({"actor": torch.zeros(3, obs_width)}, batch_size=[3])
  actor = GatedAntiFallGetUpActor(
    obs,
    {"actor": ("actor",)},
    "actor",
    output_dim=2,
    hidden_dims=(2, 2, 2),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1},
  )

  actor.walking_actor.forward = lambda obs, **kwargs: torch.full((obs["actor"].shape[0], 2), 1.0)
  actor.recovery_actor.forward = lambda obs, **kwargs: torch.full((obs["actor"].shape[0], 2), -1.0)

  # The default inferred gate uses the latest getup_progress frame:
  # [height_progress, facing_up] plus an explicit recovery-phase flag.  Upright
  # env 0 should walk, low-height env 1, low-uprightness env 2, upright
  # phase-flagged env 3 should use the recovery prior, and the moderately
  # tilted env 4 should still walk because the actor gate matches the action
  # term's fallen_tilt_threshold=0.75 rather than an earlier standalone
  # upright-success threshold.
  gate_indices = actor.recovery_gate_indices.tolist()
  assert gate_indices == [601, 602, 606]
  assert actor.recovery_gate_height_threshold == pytest.approx(0.55)
  assert actor.recovery_gate_upright_threshold == pytest.approx(math.sqrt(1.0 - 0.75**2))
  obs = TensorDict({"actor": torch.zeros(5, obs_width)}, batch_size=[5])
  obs["actor"][:, gate_indices[0]] = torch.tensor([0.95, 0.50, 0.95, 0.95, 0.95])
  obs["actor"][:, gate_indices[1]] = torch.tensor([0.95, 0.95, 0.10, 0.95, 0.70])
  obs["actor"][:, gate_indices[2]] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

  output = actor(obs)

  torch.testing.assert_close(output[0], torch.ones(2))
  torch.testing.assert_close(output[1], -torch.ones(2))
  torch.testing.assert_close(output[2], -torch.ones(2))
  torch.testing.assert_close(output[3], -torch.ones(2))
  torch.testing.assert_close(output[4], torch.ones(2))


def test_gated_antifall_getup_actor_onnx_wrapper_uses_same_gate() -> None:
  from tensordict import TensorDict

  from scripts import train
  from src.tasks.velocity.rl.gated_actor import GatedAntiFallGetUpActor

  obs_width = train._layout_width(train._g1_antifall_getup_actor_layout())
  obs = TensorDict({"actor": torch.zeros(2, obs_width)}, batch_size=[2])
  actor = GatedAntiFallGetUpActor(
    obs,
    {"actor": ("actor",)},
    "actor",
    output_dim=1,
    hidden_dims=(2,),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1},
  )
  with torch.no_grad():
    actor.walking_actor.obs_normalizer.eval()
    actor.recovery_actor.obs_normalizer.eval()
    actor.walking_actor.mlp[0].weight.zero_()
    actor.walking_actor.mlp[0].bias.zero_()
    actor.walking_actor.mlp[2].weight.zero_()
    actor.walking_actor.mlp[2].bias.fill_(1.0)
    actor.recovery_actor.mlp[0].weight.zero_()
    actor.recovery_actor.mlp[0].bias.zero_()
    actor.recovery_actor.mlp[2].weight.zero_()
    actor.recovery_actor.mlp[2].bias.fill_(-1.0)

  export_model = actor.as_onnx(verbose=False)
  x = torch.zeros(2, obs_width)
  x[:, 601] = torch.tensor([0.95, 0.95])
  x[:, 602] = torch.tensor([0.95, 0.95])
  x[:, 606] = torch.tensor([0.0, 1.0])

  output = export_model(x)

  assert export_model.get_dummy_inputs()[0].shape == (1, obs_width)
  assert export_model.input_names == ["obs"]
  assert export_model.output_names == ["actions"]
  torch.testing.assert_close(output, torch.tensor([[1.0], [-1.0]]))

  x_mid_tilt = torch.zeros(1, obs_width)
  x_mid_tilt[:, 601] = 0.95
  x_mid_tilt[:, 602] = 0.70
  x_mid_tilt[:, 606] = 0.0
  torch.testing.assert_close(export_model(x_mid_tilt), torch.tensor([[1.0]]))


def test_gated_antifall_getup_actor_updates_branch_normalizers_by_gate() -> None:
  from tensordict import TensorDict

  from scripts import train
  from src.tasks.velocity.rl.gated_actor import GatedAntiFallGetUpActor

  obs_width = train._layout_width(train._g1_antifall_getup_actor_layout())
  obs = TensorDict({"actor": torch.zeros(4, obs_width)}, batch_size=[4])
  actor = GatedAntiFallGetUpActor(
    obs,
    {"actor": ("actor",)},
    "actor",
    output_dim=1,
    hidden_dims=(2,),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1},
  )
  gate_indices = actor.recovery_gate_indices.tolist()
  obs["actor"][:, 0] = torch.tensor([10.0, 20.0, 100.0, 200.0])
  obs["actor"][:, gate_indices[0]] = torch.tensor([0.95, 0.95, 0.50, 0.95])
  obs["actor"][:, gate_indices[1]] = torch.tensor([0.95, 0.95, 0.95, 0.10])
  obs["actor"][:, gate_indices[2]] = torch.tensor([0.0, 0.0, 0.0, 0.0])

  actor.update_normalization(obs)

  assert actor.walking_actor.obs_normalizer.count.item() == 2
  assert actor.recovery_actor.obs_normalizer.count.item() == 2
  torch.testing.assert_close(actor.walking_actor.obs_normalizer.mean[0], torch.tensor(15.0))
  torch.testing.assert_close(actor.recovery_actor.obs_normalizer.mean[0], torch.tensor(150.0))


def test_load_fused_antifall_getup_actor_loads_gated_policy_branches(tmp_path) -> None:
  from tensordict import TensorDict

  from scripts import train
  from src.tasks.velocity.rl.gated_actor import GatedAntiFallGetUpActor

  stage_width = train._layout_width(train._g1_antifall_stage_actor_layout())
  target_width = train._layout_width(train._g1_antifall_getup_actor_layout())
  projection = train._build_observation_projection(
    train._g1_antifall_stage_actor_layout(),
    train._g1_antifall_getup_actor_layout(),
  )
  walking_weight_col = next(
    col for col, source_col in enumerate(projection.weight_source_by_target) if source_col is not None
  )
  walking_source_col = projection.weight_source_by_target[walking_weight_col]
  recovery_only_col = next(
    col for col, source_col in enumerate(projection.stats_source_by_target) if source_col is None
  )
  assert walking_source_col is not None

  obs = TensorDict({"actor": torch.zeros(1, target_width)}, batch_size=[1])
  policy = GatedAntiFallGetUpActor(
    obs,
    {"actor": ("actor",)},
    "actor",
    output_dim=29,
    hidden_dims=(2, 3, 4),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
  )

  def _filled_like(state: dict[str, torch.Tensor], fill: float) -> dict[str, torch.Tensor]:
    return {
      key: torch.full_like(value, fill) if torch.is_tensor(value) and value.is_floating_point() else value.clone()
      for key, value in state.items()
    }

  walking_state = _filled_like(policy.walking_actor.state_dict(), 1.0)
  recovery_state = _filled_like(policy.recovery_actor.state_dict(), 2.0)
  walking_state["obs_normalizer._mean"] = torch.full((1, stage_width), 1.0)
  walking_state["obs_normalizer._var"] = torch.full((1, stage_width), 1.0)
  walking_state["obs_normalizer._std"] = torch.full((1, stage_width), 1.0)
  walking_state["mlp.0.weight"] = torch.arange(
    walking_state["mlp.0.weight"].shape[0] * stage_width,
    dtype=torch.float32,
  ).reshape(walking_state["mlp.0.weight"].shape[0], stage_width)
  walking_state["distribution.std_param"] = torch.full((29,), 0.11)
  recovery_state["mlp.0.weight"] = torch.arange(
    recovery_state["mlp.0.weight"].numel(),
    dtype=torch.float32,
  ).reshape_as(recovery_state["mlp.0.weight"]) + 1000.0

  walking_checkpoint = tmp_path / "walking.pt"
  recovery_checkpoint = tmp_path / "recovery.pt"
  torch.save({"actor_state_dict": walking_state}, walking_checkpoint)
  torch.save({"actor_state_dict": recovery_state}, recovery_checkpoint)

  class _FakeAlg:
    def get_policy(self):
      return policy

  class _FakeRunner:
    alg = _FakeAlg()

  train._load_fused_antifall_getup_actor(
    _FakeRunner(),
    walking_resume_path=walking_checkpoint,
    recovery_resume_path=recovery_checkpoint,
    map_location="cpu",
  )

  torch.testing.assert_close(
    policy.walking_actor.state_dict()["mlp.0.weight"][:, walking_weight_col],
    walking_state["mlp.0.weight"][:, walking_source_col],
  )
  torch.testing.assert_close(
    policy.recovery_actor.state_dict()["mlp.0.weight"][:, recovery_only_col],
    recovery_state["mlp.0.weight"][:, recovery_only_col],
  )
  torch.testing.assert_close(policy.distribution.std_param, torch.full((29,), 0.11))


def test_recovery_action_output_scale_infers_legacy_getup_scale_for_antifall_target(tmp_path) -> None:
  from scripts import train

  checkpoint = tmp_path / "model.pt"
  state = {"mlp.0.weight": torch.zeros(2, train._layout_width(train._g1_getup_actor_layout()))}
  torch.save({"actor_state_dict": state}, checkpoint)
  target_env = train.TrainConfig.from_task("Unitree-G1-AntiFall-GetUp").env

  assert train._recovery_action_output_scale_for_checkpoint(
    checkpoint,
    target_env,
    checkpoint_state=state,
    map_location="cpu",
  ) == pytest.approx(4.0)


def test_recovery_action_output_scale_uses_saved_antifall_scale_without_double_rescale(tmp_path) -> None:
  from scripts import train

  run_dir = tmp_path / "run"
  params_dir = run_dir / "params"
  params_dir.mkdir(parents=True)
  checkpoint = run_dir / "model.pt"
  state = {"mlp.0.weight": torch.zeros(2, train._layout_width(train._g1_antifall_getup_actor_layout()))}
  torch.save({"actor_state_dict": state}, checkpoint)
  (params_dir / "env.yaml").write_text(
    "actions:\n"
    "  joint_pos:\n"
    "    recovery_action_scale: 0.25\n",
  )
  target_env = train.TrainConfig.from_task("Unitree-G1-AntiFall-GetUp").env

  assert train._recovery_action_output_scale_for_checkpoint(
    checkpoint,
    target_env,
    checkpoint_state=state,
    map_location="cpu",
  ) == pytest.approx(1.0)


def test_load_fused_antifall_getup_actor_rescales_legacy_recovery_branch_delta_contract(tmp_path) -> None:
  from tensordict import TensorDict

  from scripts import train
  from src.tasks.velocity.rl.gated_actor import GatedAntiFallGetUpActor

  target_width = train._layout_width(train._g1_antifall_getup_actor_layout())
  obs = TensorDict({"actor": torch.zeros(1, target_width)}, batch_size=[1])
  policy = GatedAntiFallGetUpActor(
    obs,
    {"actor": ("actor",)},
    "actor",
    output_dim=29,
    hidden_dims=(2, 3, 4),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
  )

  walking_state = {
    key: torch.zeros_like(value) if torch.is_tensor(value) and value.is_floating_point() else value.clone()
    for key, value in policy.walking_actor.state_dict().items()
  }
  recovery_state = {
    key: torch.zeros_like(value) if torch.is_tensor(value) and value.is_floating_point() else value.clone()
    for key, value in policy.recovery_actor.state_dict().items()
  }
  getup_width = train._layout_width(train._g1_getup_actor_layout())
  recovery_state["mlp.0.weight"] = torch.zeros(recovery_state["mlp.0.weight"].shape[0], getup_width)
  recovery_state["obs_normalizer._mean"] = torch.zeros(1, getup_width)
  recovery_state["obs_normalizer._var"] = torch.ones(1, getup_width)
  recovery_state["obs_normalizer._std"] = torch.ones(1, getup_width)
  recovery_state["mlp.6.weight"] = torch.arange(23 * 4, dtype=torch.float32).reshape(23, 4)
  recovery_state["mlp.6.bias"] = torch.arange(23, dtype=torch.float32)
  recovery_state["distribution.std_param"] = torch.full((23,), 0.25)

  walking_checkpoint = tmp_path / "walking.pt"
  recovery_checkpoint = tmp_path / "recovery.pt"
  torch.save({"actor_state_dict": walking_state}, walking_checkpoint)
  torch.save({"actor_state_dict": recovery_state}, recovery_checkpoint)

  class _FakeAlg:
    def get_policy(self):
      return policy

  class _FakeRunner:
    alg = _FakeAlg()

  train._load_fused_antifall_getup_actor(
    _FakeRunner(),
    walking_resume_path=walking_checkpoint,
    recovery_resume_path=recovery_checkpoint,
    target_env_cfg=train.TrainConfig.from_task("Unitree-G1-AntiFall-GetUp").env,
    map_location="cpu",
  )

  loaded = policy.recovery_actor.state_dict()
  torch.testing.assert_close(loaded["mlp.6.weight"][0], torch.arange(4, dtype=torch.float32) * 4.0)
  torch.testing.assert_close(loaded["mlp.6.bias"][1], torch.tensor(4.0))
  torch.testing.assert_close(loaded["distribution.std_param"][0], torch.tensor(1.0))
  torch.testing.assert_close(loaded["distribution.std_param"][13], torch.tensor(0.5))


def test_expand_legacy_antifall_getup_actor_adds_neutral_recovery_phase_column() -> None:
  from scripts import train

  source_layout = train._g1_antifall_getup_actor_layout_without_recovery_phase()
  target_layout = train._g1_antifall_getup_actor_layout()
  source_width = train._layout_width(source_layout)
  target_width = train._layout_width(target_layout)
  projection = train._build_observation_projection(source_layout, target_layout)
  phase_col = next(
    idx
    for idx, source_idx in enumerate(projection.stats_source_by_target)
    if source_idx is None
  )
  height_col = next(
    idx
    for idx, source_idx in enumerate(projection.weight_source_by_target)
    if source_idx is not None and idx > phase_col
  )
  height_source_col = projection.weight_source_by_target[height_col]
  assert source_width == 2176
  assert target_width == 2177
  assert phase_col == 606
  assert height_source_col is not None

  checkpoint_state = {
    "obs_normalizer._mean": torch.arange(source_width, dtype=torch.float32).reshape(1, source_width),
    "obs_normalizer._var": torch.arange(source_width, dtype=torch.float32).reshape(1, source_width) + 100.0,
    "obs_normalizer._std": torch.arange(source_width, dtype=torch.float32).reshape(1, source_width) + 200.0,
    "mlp.0.weight": torch.arange(2 * source_width, dtype=torch.float32).reshape(2, source_width),
  }
  target_state = {
    "obs_normalizer._mean": torch.zeros(1, target_width),
    "obs_normalizer._var": torch.ones(1, target_width),
    "obs_normalizer._std": torch.ones(1, target_width),
    "mlp.0.weight": torch.full((2, target_width), -7.0),
  }

  assert train._expand_model_input_state(checkpoint_state, target_state) is True
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, phase_col], torch.zeros(1))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._var"][:, phase_col], torch.ones(1))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._std"][:, phase_col], torch.ones(1))
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, phase_col], torch.zeros(2))
  torch.testing.assert_close(
    checkpoint_state["mlp.0.weight"][:, height_col],
    torch.arange(2 * source_width, dtype=torch.float32).reshape(2, source_width)[:, height_source_col],
  )


def test_fuse_antifall_getup_actor_state_keeps_walking_columns_and_recovers_getup_columns() -> None:
  from scripts import train

  stage_layout = train._g1_antifall_stage_actor_layout()
  target_layout = train._g1_antifall_getup_actor_layout()
  stage_width = train._layout_width(stage_layout)
  target_width = train._layout_width(target_layout)
  projection = train._build_observation_projection(stage_layout, target_layout)

  walking_weight_col = next(
    col for col, source_col in enumerate(projection.weight_source_by_target) if source_col is not None
  )
  walking_weight_source_col = projection.weight_source_by_target[walking_weight_col]
  recovery_weight_col = next(
    col for col, source_col in enumerate(projection.stats_source_by_target) if source_col is None
  )
  extra_common_history_col = next(
    col
    for col, source_col in enumerate(projection.weight_source_by_target)
    if source_col is None and projection.stats_source_by_target[col] is not None
  )
  walking_stats_col = next(
    col for col, source_col in enumerate(projection.stats_source_by_target) if source_col is not None
  )
  walking_stats_source_col = projection.stats_source_by_target[walking_stats_col]
  recovery_stats_col = next(
    col for col, source_col in enumerate(projection.stats_source_by_target) if source_col is None
  )

  walking_first_layer = torch.arange(2 * stage_width, dtype=torch.float32).reshape(2, stage_width) + 100.0
  recovery_first_layer = torch.arange(2 * target_width, dtype=torch.float32).reshape(2, target_width) + 200.0
  walking_state = {
    "obs_normalizer._mean": torch.arange(stage_width, dtype=torch.float32).reshape(1, stage_width) + 10.0,
    "obs_normalizer._var": torch.arange(stage_width, dtype=torch.float32).reshape(1, stage_width) + 20.0,
    "obs_normalizer._std": torch.arange(stage_width, dtype=torch.float32).reshape(1, stage_width) + 30.0,
    "obs_normalizer.count": torch.tensor(123.0),
    "distribution.std_param": torch.full((29,), 0.11),
    "mlp.0.weight": walking_first_layer.clone(),
    "mlp.0.bias": torch.full((2,), 1.0),
    "mlp.2.weight": torch.full((3, 2), 2.0),
    "mlp.2.bias": torch.full((3,), 3.0),
    "mlp.4.weight": torch.full((4, 3), 4.0),
    "mlp.4.bias": torch.full((4,), 5.0),
    "mlp.6.weight": torch.full((29, 4), 6.0),
    "mlp.6.bias": torch.full((29,), 7.0),
  }
  recovery_state = {
    "obs_normalizer._mean": torch.arange(target_width, dtype=torch.float32).reshape(1, target_width) + 1000.0,
    "obs_normalizer._var": torch.arange(target_width, dtype=torch.float32).reshape(1, target_width) + 2000.0,
    "obs_normalizer._std": torch.arange(target_width, dtype=torch.float32).reshape(1, target_width) + 3000.0,
    "obs_normalizer.count": torch.tensor(456.0),
    "distribution.std_param": torch.full((29,), 0.22),
    "mlp.0.weight": recovery_first_layer.clone(),
    "mlp.0.bias": torch.full((2,), 11.0),
    "mlp.2.weight": torch.full((3, 2), 12.0),
    "mlp.2.bias": torch.full((3,), 13.0),
    "mlp.4.weight": torch.full((4, 3), 14.0),
    "mlp.4.bias": torch.full((4,), 15.0),
    "mlp.6.weight": torch.full((29, 4), 16.0),
    "mlp.6.bias": torch.full((29,), 17.0),
  }
  target_state = {
    "obs_normalizer._mean": torch.zeros(1, target_width),
    "obs_normalizer._var": torch.ones(1, target_width),
    "obs_normalizer._std": torch.ones(1, target_width),
    "obs_normalizer.count": torch.tensor(0.0),
    "distribution.std_param": torch.ones(29),
    "mlp.0.weight": torch.zeros(2, target_width),
    "mlp.0.bias": torch.zeros(2),
    "mlp.2.weight": torch.zeros(3, 2),
    "mlp.2.bias": torch.zeros(3),
    "mlp.4.weight": torch.zeros(4, 3),
    "mlp.4.bias": torch.zeros(4),
    "mlp.6.weight": torch.zeros(29, 4),
    "mlp.6.bias": torch.zeros(29),
  }

  fused = train._fuse_antifall_getup_actor_state(walking_state, recovery_state, target_state)

  assert fused["mlp.0.weight"].shape == (2, target_width)
  assert walking_weight_source_col is not None
  assert walking_stats_source_col is not None
  torch.testing.assert_close(
    fused["mlp.0.weight"][:, walking_weight_col],
    walking_first_layer[:, walking_weight_source_col],
  )
  torch.testing.assert_close(
    fused["mlp.0.weight"][:, recovery_weight_col],
    recovery_first_layer[:, recovery_weight_col],
  )
  torch.testing.assert_close(
    fused["mlp.0.weight"][:, extra_common_history_col],
    torch.zeros(2),
  )
  torch.testing.assert_close(
    fused["obs_normalizer._mean"][:, walking_stats_col],
    walking_state["obs_normalizer._mean"][:, walking_stats_source_col],
  )
  torch.testing.assert_close(
    fused["obs_normalizer._mean"][:, recovery_stats_col],
    recovery_state["obs_normalizer._mean"][:, recovery_stats_col],
  )
  torch.testing.assert_close(fused["mlp.2.weight"], walking_state["mlp.2.weight"])
  torch.testing.assert_close(fused["mlp.6.weight"], walking_state["mlp.6.weight"])
  torch.testing.assert_close(fused["distribution.std_param"], walking_state["distribution.std_param"])


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


def test_expand_actor_checkpoint_input_projects_getup_terms_for_antifall_resume() -> None:
  from scripts.train import _expand_model_input_state

  checkpoint_state = {
    "obs_normalizer._mean": torch.arange(1978, dtype=torch.float32).reshape(1, 1978),
    "obs_normalizer._var": torch.arange(1978, dtype=torch.float32).reshape(1, 1978) + 10.0,
    "obs_normalizer._std": torch.arange(1978, dtype=torch.float32).reshape(1, 1978) + 20.0,
    "distribution.std_param": torch.full((23,), 0.25),
    "mlp.0.weight": torch.arange(512 * 1978, dtype=torch.float32).reshape(512, 1978),
    "mlp.0.bias": torch.arange(512, dtype=torch.float32),
    "mlp.6.weight": torch.arange(23 * 4, dtype=torch.float32).reshape(23, 4),
    "mlp.6.bias": torch.arange(23, dtype=torch.float32),
  }
  original_first_layer = checkpoint_state["mlp.0.weight"].clone()
  target_state = {
    "obs_normalizer._mean": torch.zeros(1, 2177),
    "obs_normalizer._var": torch.ones(1, 2177),
    "obs_normalizer._std": torch.ones(1, 2177),
    "distribution.std_param": torch.full((29,), 0.5),
    "mlp.0.weight": torch.full((512, 2177), -7.0),
    "mlp.0.bias": torch.full((512,), -3.0),
    "mlp.6.weight": torch.full((29, 4), -9.0),
    "mlp.6.bias": torch.full((29,), -3.0),
  }

  expanded = _expand_model_input_state(checkpoint_state, target_state)

  assert expanded is True
  assert checkpoint_state["obs_normalizer._mean"].shape == (1, 2177)
  assert checkpoint_state["mlp.0.weight"].shape == (512, 2177)

  # AntiFall-GetUp keeps GetUp's six-frame recovery history and height scan,
  # while widening joint/body features to the 29-DoF walking robot. Compatible
  # resume must therefore align by term/feature name and preserve matching
  # history frames rather than blindly truncating or prefix-copying columns.
  torch.testing.assert_close(
    checkpoint_state["obs_normalizer._mean"][:, 0:18],
    torch.arange(18, dtype=torch.float32).reshape(1, 18),
  )
  torch.testing.assert_close(
    checkpoint_state["obs_normalizer._var"][:, 0:18],
    torch.arange(18, dtype=torch.float32).reshape(1, 18) + 10.0,
  )
  torch.testing.assert_close(
    checkpoint_state["obs_normalizer._std"][:, 0:18],
    torch.arange(18, dtype=torch.float32).reshape(1, 18) + 20.0,
  )
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 0:18], original_first_layer[:, 0:18])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 54], original_first_layer[:, 54])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 66], original_first_layer[:, 66])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 67], torch.zeros(512))
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 69], original_first_layer[:, 67])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 576:591], original_first_layer[:, 468:483])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 606], torch.zeros(512))
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 607], original_first_layer[:, 498])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 1055], original_first_layer[:, 856])
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, 606], torch.tensor([0.0]))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, 607], torch.tensor([498.0]))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, 1055], torch.tensor([856.0]))
  # The 23-DoF GetUp XML names the hand endpoint as *_wrist_roll_rubber_hand,
  # while the 29-DoF AntiFall XML carries the rubber hand geometry under
  # *_wrist_yaw_link.  The transfer must preserve those endpoint body-state
  # features by semantic alias rather than treating them as missing bodies.
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, 671], torch.tensor([550.0]))
  torch.testing.assert_close(checkpoint_state["obs_normalizer._mean"][:, 692], torch.tensor([565.0]))
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 671], original_first_layer[:, 550])
  torch.testing.assert_close(checkpoint_state["mlp.0.weight"][:, 692], original_first_layer[:, 565])
  torch.testing.assert_close(
    checkpoint_state["obs_normalizer._mean"][:, -1122:],
    torch.arange(856, 1978, dtype=torch.float32).reshape(1, 1122),
  )
  assert checkpoint_state["distribution.std_param"].shape == (29,)
  assert checkpoint_state["mlp.6.weight"].shape == (29, 4)
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][0], torch.arange(4, dtype=torch.float32))
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][13], torch.zeros(4))
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][15], torch.arange(13 * 4, 14 * 4, dtype=torch.float32))
  assert checkpoint_state["distribution.std_param"][13] == pytest.approx(0.5)


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


def test_expand_getup_actor_output_rescales_recovery_delta_contract_without_touching_new_joints() -> None:
  from scripts import train

  checkpoint_state = {
    "distribution.std_param": torch.full((23,), 0.25),
    "mlp.0.weight": torch.zeros(2, 1978),
    "mlp.6.weight": torch.arange(23 * 4, dtype=torch.float32).reshape(23, 4),
    "mlp.6.bias": torch.arange(23, dtype=torch.float32),
  }
  target_state = {
    "distribution.std_param": torch.full((29,), 0.5),
    "mlp.0.weight": torch.zeros(2, 2177),
    "mlp.6.weight": torch.full((29, 4), -9.0),
    "mlp.6.bias": torch.full((29,), -3.0),
  }

  changed = train._expand_model_input_state(
    checkpoint_state,
    target_state,
    action_output_scale=4.0,
  )

  assert changed is True
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][0], torch.arange(4, dtype=torch.float32) * 4.0)
  torch.testing.assert_close(checkpoint_state["mlp.6.bias"][0], torch.tensor(0.0))
  torch.testing.assert_close(checkpoint_state["distribution.std_param"][0], torch.tensor(1.0))
  # The 29-DoF-only joints are not present in the source GetUp actor.  Keep
  # their neutral/default target initialization instead of multiplying the
  # target actor's default std or injecting arbitrary output rows.
  torch.testing.assert_close(checkpoint_state["mlp.6.weight"][13], torch.zeros(4))
  torch.testing.assert_close(checkpoint_state["mlp.6.bias"][13], torch.tensor(0.0))
  torch.testing.assert_close(checkpoint_state["distribution.std_param"][13], torch.tensor(0.5))
  assert checkpoint_state["mlp.6.bias"][13] == pytest.approx(0.0)
