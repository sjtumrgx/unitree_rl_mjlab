from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from src.tasks.velocity.rl.antifall_curriculum import CURRICULUM_TASK_ID
from src.tasks.velocity.rl.curriculum_runner import AntiFallCurriculumRunner


class DummyVecEnv:
  def __init__(self, cfg):
    self.cfg = cfg
    self.render_mode = None
    self.env = None
    self.closed = False

  def close(self):
    self.closed = True


class FakeChildEnv:
  def __init__(self):
    self.closed = False

  def close(self):
    self.closed = True


class FakeChildRunner:
  def __init__(self):
    self.env = FakeChildEnv()
    self.current_learning_iteration = 0
    self.is_distributed = False
    self.gpu_global_rank = 0


def test_curriculum_runner_orchestrates_stage_order_and_checkpoint_lineage(monkeypatch, tmp_path: Path) -> None:
  env_cfg = load_env_cfg(CURRICULUM_TASK_ID)
  rl_cfg = load_rl_cfg(CURRICULUM_TASK_ID)
  runner = AntiFallCurriculumRunner(DummyVecEnv(env_cfg), asdict(rl_cfg), str(tmp_path), "cpu")

  built: list[tuple[int, str]] = []
  loaded: list[tuple[int, Path | None, bool]] = []
  run_calls: list[tuple[int, str, str]] = []

  def fake_build_stage_runner(self, *, stage_index, task_id, stage_name, stage_dir, reuse_bootstrap):
    built.append((stage_index, task_id))
    return FakeChildRunner()

  def fake_load_child_runner(self, *, child_runner, stage_index, previous_checkpoint, resume_current_stage):
    loaded.append((stage_index, previous_checkpoint, resume_current_stage))
    return "fresh" if previous_checkpoint is None else "full"

  def fake_run_stage(self, *, child_runner, stage_index, stage_name, stage_dir, load_mode, init_at_random_ep_len):
    run_calls.append((stage_index, stage_name, load_mode))
    checkpoint = stage_dir / f"model_{stage_index + 1}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")
    self._update_stage_record(
      stage_index=stage_index,
      status="completed",
      load_mode=load_mode,
      latest_checkpoint=checkpoint,
      stage_iteration=stage_index + 1,
      promotion_reason="forced_for_test",
      metrics={"iteration": stage_index + 1},
      is_primary=True,
    )
    self.current_learning_iteration += 1
    return {
      "status": "completed",
      "latest_checkpoint": checkpoint,
      "failure_reason": None,
      "promotion_reason": "forced_for_test",
    }

  monkeypatch.setattr(AntiFallCurriculumRunner, "_build_stage_runner", fake_build_stage_runner)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_load_child_runner", fake_load_child_runner)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_run_stage", fake_run_stage)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_save_root_checkpoint", lambda self: None)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_copy_latest_policy", lambda self, stage_dir: None)

  runner.learn(2)

  assert [task_id for _, task_id in built] == list(rl_cfg.curriculum.stage_task_ids)
  assert loaded[0][1] is None
  assert loaded[1][1] is not None
  assert run_calls[0][1] == "stage0"
  assert run_calls[-1][1] == "stage4b"
  assert runner._manifest["completed_stage_indices"] == [0, 1, 2, 3, 4, 5]
  assert runner._manifest["latest_checkpoint"].endswith("stages/05_stage4b/model_6.pt")


def test_curriculum_runner_resume_skips_completed_stages(monkeypatch, tmp_path: Path) -> None:
  env_cfg = load_env_cfg(CURRICULUM_TASK_ID)
  rl_cfg = load_rl_cfg(CURRICULUM_TASK_ID)
  runner = AntiFallCurriculumRunner(DummyVecEnv(env_cfg), asdict(rl_cfg), str(tmp_path), "cpu")

  manifest = runner._manifest
  manifest["completed_stage_indices"] = [0, 1]
  manifest["active_stage_index"] = 2
  manifest["stages"][0]["status"] = "completed"
  manifest["stages"][1]["status"] = "completed"
  manifest["stages"][2]["status"] = "in_progress"
  checkpoint = tmp_path / "stages" / "02_stage2" / "model_5.pt"
  checkpoint.parent.mkdir(parents=True, exist_ok=True)
  checkpoint.write_text("checkpoint")
  manifest["latest_checkpoint"] = runner._relative_to_run(checkpoint)
  runner._manifest = manifest
  runner._resume_state = {
    "active_stage_index": 2,
    "latest_stage_checkpoint": runner._relative_to_run(checkpoint),
  }
  runner._loaded = True

  built: list[int] = []
  loaded: list[tuple[int, Path | None, bool]] = []

  def fake_build_stage_runner(self, *, stage_index, task_id, stage_name, stage_dir, reuse_bootstrap):
    built.append(stage_index)
    return FakeChildRunner()

  def fake_load_child_runner(self, *, child_runner, stage_index, previous_checkpoint, resume_current_stage):
    loaded.append((stage_index, previous_checkpoint, resume_current_stage))
    return "full_resume"

  def fake_run_stage(self, *, child_runner, stage_index, stage_name, stage_dir, load_mode, init_at_random_ep_len):
    checkpoint_path = stage_dir / f"model_{stage_index + 10}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("checkpoint")
    self._update_stage_record(
      stage_index=stage_index,
      status="completed",
      load_mode=load_mode,
      latest_checkpoint=checkpoint_path,
      stage_iteration=stage_index + 10,
      promotion_reason="forced_for_test",
      metrics={"iteration": stage_index + 10},
      is_primary=True,
    )
    return {
      "status": "completed",
      "latest_checkpoint": checkpoint_path,
      "failure_reason": None,
      "promotion_reason": "forced_for_test",
    }

  monkeypatch.setattr(AntiFallCurriculumRunner, "_build_stage_runner", fake_build_stage_runner)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_load_child_runner", fake_load_child_runner)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_run_stage", fake_run_stage)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_save_root_checkpoint", lambda self: None)
  monkeypatch.setattr(AntiFallCurriculumRunner, "_copy_latest_policy", lambda self, stage_dir: None)

  runner.learn(3)

  assert built[0] == 2
  assert loaded[0][2] is True
  assert all(stage_index >= 2 for stage_index in built)


def test_curriculum_runner_finishes_stage_saves_before_stopping_logger(
  monkeypatch, tmp_path: Path
) -> None:
  env_cfg = load_env_cfg(CURRICULUM_TASK_ID)
  rl_cfg = load_rl_cfg(CURRICULUM_TASK_ID)
  runner = AntiFallCurriculumRunner(DummyVecEnv(env_cfg), asdict(rl_cfg), str(tmp_path), "cpu")
  runner.curriculum_cfg.per_stage_max_iterations = 0

  events: list[str] = []
  stage_dir = tmp_path / "stages" / "00_stage0"
  stage_dir.mkdir(parents=True, exist_ok=True)

  class RecordingLogger:
    def init_logging_writer(self) -> None:
      events.append("init")

    def stop_logging_writer(self) -> None:
      events.append("stop")

  class RecordingAlg:
    def train_mode(self) -> None:
      events.append("train_mode")

  class RecordingEnv:
    def __init__(self) -> None:
      self.device = "cpu"
      self.max_episode_length = 1
      self.episode_length_buf = torch.zeros((1, 1), dtype=torch.long)
      self.unwrapped = SimpleNamespace(
        metrics_manager=SimpleNamespace(active_terms=None, _step_values={}),
        common_step_counter=0,
      )

    def get_observations(self) -> torch.Tensor:
      return torch.zeros((1, 1), dtype=torch.float32)

  class RecordingChildRunner:
    def __init__(self) -> None:
      self.env = RecordingEnv()
      self.alg = RecordingAlg()
      self.logger = RecordingLogger()
      self.cfg = {"num_steps_per_env": 1, "algorithm": {"rnd_cfg": False}}
      self.current_learning_iteration = 0
      self.is_distributed = False
      self.gpu_global_rank = 0

  def fake_save_stage_checkpoint(self, *, child_runner, stage_dir):
    del child_runner
    events.append("stage_save")
    checkpoint = stage_dir / "model_0.pt"
    checkpoint.write_text("checkpoint")
    return checkpoint

  def fake_copy_latest_policy(self, stage_dir, *, child_runner):
    del stage_dir, child_runner
    events.append("copy_policy")

  def fake_save_root_checkpoint(self, child_runner):
    del child_runner
    events.append("root_save")

  def fake_distributed_barrier(self, child_runner):
    del child_runner
    events.append("barrier")

  monkeypatch.setattr(
    AntiFallCurriculumRunner, "_save_stage_checkpoint", fake_save_stage_checkpoint
  )
  monkeypatch.setattr(
    AntiFallCurriculumRunner, "_copy_latest_policy", fake_copy_latest_policy
  )
  monkeypatch.setattr(
    AntiFallCurriculumRunner, "_save_root_checkpoint", fake_save_root_checkpoint
  )
  monkeypatch.setattr(
    AntiFallCurriculumRunner, "_distributed_barrier", fake_distributed_barrier
  )

  result = runner._run_stage(
    child_runner=RecordingChildRunner(),
    stage_index=0,
    stage_name="stage0",
    stage_dir=stage_dir,
    load_mode="fresh",
    init_at_random_ep_len=False,
  )

  assert result["status"] == "completed"
  assert result["promotion_reason"] == "max_iterations_reached"
  assert events == [
    "train_mode",
    "init",
    "stage_save",
    "copy_policy",
    "root_save",
    "barrier",
    "stop",
  ]
