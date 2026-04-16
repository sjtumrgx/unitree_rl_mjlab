"""Curriculum runner for the Unitree G1 anti-fall task family."""

from __future__ import annotations

import shutil
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from rsl_rl.utils import check_nan

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.wrappers import VideoRecorder

from .antifall_curriculum import (
  ANTI_FALL_STAGE_TASK_IDS,
  CURRICULUM_TASK_ID,
  AntiFallCurriculumCfg,
  curriculum_stage_name,
)
from .curriculum_metrics import (
  StagePromotionMonitor,
  evaluate_promotion,
  metrics_from_totals,
)
from .curriculum_state import (
  load_manifest,
  manifest_for_update,
  new_curriculum_manifest,
  transition_load_mode,
  write_manifest,
)
from .runner import VelocityOnPolicyRunner


class AntiFallCurriculumRunner:
  """Run the anti-fall stages as one top-level curriculum process.

  The runner intentionally rebuilds stage envs/runners at stage boundaries instead
  of mutating a live environment in-place. This keeps the curriculum path aligned
  with the existing single-task training stack while still presenting a single
  top-level command/process to the user.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.env = env
    self.cfg = deepcopy(train_cfg)
    self.device = device
    self.log_dir = Path(log_dir) if log_dir is not None else None
    self.current_learning_iteration = 0

    curriculum_payload = deepcopy(self.cfg.get("curriculum") or {})
    if isinstance(curriculum_payload, AntiFallCurriculumCfg):
      curriculum_payload = asdict(curriculum_payload)
    self.curriculum_cfg = AntiFallCurriculumCfg(**curriculum_payload)
    self.curriculum_cfg.per_stage_max_iterations = int(
      self.cfg.get("max_iterations", self.curriculum_cfg.per_stage_max_iterations)
    )
    if not self.curriculum_cfg.stage_task_ids:
      self.curriculum_cfg.stage_task_ids = ANTI_FALL_STAGE_TASK_IDS

    self._initial_env_cfg_template = deepcopy(self.env.cfg)
    self._git_repo_paths: list[str] = []
    self._loaded = False
    self._resume_state: dict[str, Any] = {}
    self._bootstrap_env_released = False
    self._video_template = self._capture_video_template()
    self._manifest_path = (
      self.log_dir / "curriculum_manifest.json"
      if self.log_dir is not None
      else Path("curriculum_manifest.json")
    )
    self._manifest = new_curriculum_manifest(
      curriculum_task_id=CURRICULUM_TASK_ID,
      stage_task_ids=self.curriculum_cfg.stage_task_ids,
      started_at=self._now(),
    )

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    if repo_file_path not in self._git_repo_paths:
      self._git_repo_paths.append(repo_file_path)

  def save(self, path: str, infos: dict | None = None) -> None:
    curriculum_state = {
      "active_stage_index": self._manifest.get("active_stage_index", 0),
      "latest_stage_checkpoint": self._manifest.get("latest_checkpoint"),
      "manifest_path": self._relative_to_run(self._manifest_path),
      "status": self._manifest.get("status"),
    }
    payload = {
      "iter": self.current_learning_iteration,
      "infos": {
        **(infos or {}),
        "curriculum_state": curriculum_state,
      },
    }
    torch.save(payload, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    del load_cfg, strict
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    self.current_learning_iteration = int(loaded_dict.get("iter", 0))
    infos = loaded_dict.get("infos") or {}
    curriculum_state = infos.get("curriculum_state") or {}
    manifest_path = curriculum_state.get("manifest_path")
    if manifest_path is not None:
      self._manifest_path = self._resolve_run_path(manifest_path)
    if self._manifest_path.exists():
      self._manifest = load_manifest(self._manifest_path)
    self._resume_state = curriculum_state
    self._loaded = True
    return infos

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    self.curriculum_cfg.per_stage_max_iterations = int(num_learning_iterations)
    if self.log_dir is not None:
      self.log_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(self._manifest_path, self._manifest)

    resume_stage_index = int(self._resume_state.get("active_stage_index", 0))
    previous_checkpoint = self._resolve_optional_run_path(
      self._resume_state.get("latest_stage_checkpoint")
    )

    if (
      self._manifest.get("status") == "complete"
      or resume_stage_index >= len(self.curriculum_cfg.stage_task_ids)
    ):
      return

    if resume_stage_index > 0:
      self._release_bootstrap_env()

    last_child_runner: VelocityOnPolicyRunner | None = None
    for stage_index, task_id in enumerate(self.curriculum_cfg.stage_task_ids):
      if stage_index < resume_stage_index:
        continue

      stage_name = curriculum_stage_name(task_id)
      stage_dir = self._stage_log_dir(stage_index, stage_name)
      stage_dir.mkdir(parents=True, exist_ok=True)

      resume_current_stage = (
        self._loaded
        and stage_index == resume_stage_index
        and self._stage_status(stage_index) == "in_progress"
      )
      child_runner = self._build_stage_runner(
        stage_index=stage_index,
        task_id=task_id,
        stage_name=stage_name,
        stage_dir=stage_dir,
        reuse_bootstrap=(stage_index == 0 and not self._bootstrap_env_released),
      )
      last_child_runner = child_runner

      load_mode = self._load_child_runner(
        child_runner=child_runner,
        stage_index=stage_index,
        previous_checkpoint=previous_checkpoint,
        resume_current_stage=resume_current_stage,
      )

      stage_result = self._run_stage(
        child_runner=child_runner,
        stage_index=stage_index,
        stage_name=stage_name,
        stage_dir=stage_dir,
        load_mode=load_mode,
        init_at_random_ep_len=(init_at_random_ep_len and not resume_current_stage),
      )
      previous_checkpoint = stage_result["latest_checkpoint"]

      self._close_child_env(child_runner.env)
      if stage_result["status"] == "failed":
        raise RuntimeError(stage_result["failure_reason"])

    self._manifest["status"] = "complete"
    self._manifest["updated_at"] = self._now()
    if last_child_runner is None or self._is_primary_process(last_child_runner):
      write_manifest(self._manifest_path, self._manifest)
    if (
      self.log_dir is not None
      and last_child_runner is not None
      and self._is_primary_process(last_child_runner)
    ):
      self.save(str(self.log_dir / f"model_{self.current_learning_iteration}.pt"))

  def _is_primary_process(self, child_runner: VelocityOnPolicyRunner) -> bool:
    return (not child_runner.is_distributed) or child_runner.gpu_global_rank == 0

  def _distributed_barrier(self, child_runner: VelocityOnPolicyRunner) -> None:
    if child_runner.is_distributed:
      torch.distributed.barrier()

  def _reduce_monitor_totals(
    self,
    child_runner: VelocityOnPolicyRunner,
    totals: dict[str, float],
  ) -> dict[str, float]:
    if not child_runner.is_distributed:
      return totals
    keys = [
      "iteration",
      "step_count",
      "controllable_locomotion_sum",
      "disturbance_count",
      "recovery_success_count",
      "recovery_latency_sum",
      "recovery_latency_count",
    ]
    tensor = torch.tensor(
      [totals[key] for key in keys], device=self.device, dtype=torch.float64
    )
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    reduced = dict(zip(keys, tensor.tolist(), strict=True))
    reduced["iteration"] = totals["iteration"]
    return reduced

  def _build_stage_runner(
    self,
    *,
    stage_index: int,
    task_id: str,
    stage_name: str,
    stage_dir: Path,
    reuse_bootstrap: bool,
  ) -> VelocityOnPolicyRunner:
    if reuse_bootstrap:
      stage_env = self.env
    else:
      raw_env = self._build_raw_env(task_id, stage_dir=stage_dir)
      stage_env = RslRlVecEnvWrapper(raw_env, clip_actions=self.cfg.get("clip_actions"))

    child_train_cfg = self._build_stage_train_cfg(stage_name)
    child_runner = VelocityOnPolicyRunner(
      stage_env, child_train_cfg, str(stage_dir), self.device
    )
    for repo_path in self._git_repo_paths:
      child_runner.add_git_repo_to_log(repo_path)

    if self.log_dir is not None and self._is_primary_process(child_runner):
      dump_yaml(stage_dir / "params" / "env.yaml", asdict(stage_env.cfg))
      dump_yaml(stage_dir / "params" / "agent.yaml", child_train_cfg)

    manifest = manifest_for_update(self._manifest)
    stage_record = manifest["stages"][stage_index]
    stage_record["status"] = "in_progress"
    stage_record["started_at"] = stage_record["started_at"] or self._now()
    stage_record["stage_iteration"] = int(stage_record.get("stage_iteration") or 0)
    stage_record["global_iteration"] = self.current_learning_iteration
    manifest["active_stage_index"] = stage_index
    manifest["status"] = "running"
    manifest["updated_at"] = self._now()
    self._manifest = manifest
    if self._is_primary_process(child_runner):
      write_manifest(self._manifest_path, self._manifest)
    return child_runner

  def _build_stage_train_cfg(self, stage_name: str) -> dict[str, Any]:
    child_cfg = deepcopy(self.cfg)
    child_cfg.pop("curriculum", None)
    child_cfg["run_name"] = stage_name
    child_cfg["resume"] = False
    child_cfg["max_iterations"] = int(self.curriculum_cfg.per_stage_max_iterations)
    return child_cfg

  def _build_raw_env(self, task_id: str, *, stage_dir: Path) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = self._initial_env_cfg_template.scene.num_envs
    cfg.seed = self._initial_env_cfg_template.seed
    cfg.sim.nan_guard = deepcopy(self._initial_env_cfg_template.sim.nan_guard)
    render_mode = self.env.render_mode
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=self.device, render_mode=render_mode)
    if self._video_template is not None and self.log_dir is not None:
      raw_env = VideoRecorder(
        raw_env,
        video_folder=stage_dir / "videos" / "train",
        episode_trigger=self._video_template["episode_trigger"],
        step_trigger=self._video_template["step_trigger"],
        video_length=self._video_template["video_length"],
        name_prefix=self._video_template["name_prefix"],
        disable_logger=self._video_template["disable_logger"],
      )
    return raw_env

  def _capture_video_template(self) -> dict[str, Any] | None:
    raw_env = getattr(self.env, "env", None)
    if not isinstance(raw_env, VideoRecorder):
      return None
    return {
      "episode_trigger": raw_env.episode_trigger,
      "step_trigger": raw_env.step_trigger,
      "video_length": raw_env.video_length,
      "name_prefix": raw_env.name_prefix,
      "disable_logger": raw_env.disable_logger,
    }

  def _load_child_runner(
    self,
    *,
    child_runner: VelocityOnPolicyRunner,
    stage_index: int,
    previous_checkpoint: Path | None,
    resume_current_stage: bool,
  ) -> str:
    is_primary = self._is_primary_process(child_runner)
    if previous_checkpoint is None:
      self._record_stage_source(stage_index, None, "fresh", is_primary=is_primary)
      return "fresh"
    if resume_current_stage:
      child_runner.load(str(previous_checkpoint), map_location=self.device)
      self._record_stage_source(
        stage_index, previous_checkpoint, "full_resume", is_primary=is_primary
      )
      return "full_resume"

    if self.curriculum_cfg.load_mode == "auto":
      source_task_id = self.curriculum_cfg.stage_task_ids[max(stage_index - 1, 0)]
      target_task_id = self.curriculum_cfg.stage_task_ids[stage_index]
      load_mode = transition_load_mode(source_task_id, target_task_id)
    else:
      load_mode = self.curriculum_cfg.load_mode

    if load_mode == "fresh":
      self._record_stage_source(
        stage_index, previous_checkpoint, load_mode, is_primary=is_primary
      )
      return load_mode

    if load_mode == "actor_only":
      child_runner.load(
        str(previous_checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=self.device,
      )
      child_runner.env.unwrapped.common_step_counter = 0
      child_runner.current_learning_iteration = 0
      self._record_stage_source(
        stage_index, previous_checkpoint, load_mode, is_primary=is_primary
      )
      return load_mode

    child_runner.load(
      str(previous_checkpoint),
      load_cfg={
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": True,
      },
      strict=True,
      map_location=self.device,
    )
    child_runner.env.unwrapped.common_step_counter = 0
    child_runner.current_learning_iteration = 0
    self._record_stage_source(
      stage_index, previous_checkpoint, load_mode, is_primary=is_primary
    )
    return load_mode

  def _run_stage(
    self,
    *,
    child_runner: VelocityOnPolicyRunner,
    stage_index: int,
    stage_name: str,
    stage_dir: Path,
    load_mode: str,
    init_at_random_ep_len: bool,
  ) -> dict[str, Any]:
    stage_budget = int(self.curriculum_cfg.per_stage_max_iterations)
    monitor = StagePromotionMonitor(stage_name, self.curriculum_cfg)
    stage_start_iteration = child_runner.current_learning_iteration
    latest_checkpoint = self._resolve_optional_run_path(
      self._manifest["stages"][stage_index].get("latest_checkpoint")
    )

    if init_at_random_ep_len and child_runner.current_learning_iteration == 0:
      child_runner.env.episode_length_buf = torch.randint_like(
        child_runner.env.episode_length_buf,
        high=int(child_runner.env.max_episode_length),
      )

    obs = child_runner.env.get_observations().to(self.device)
    child_runner.alg.train_mode()
    if child_runner.is_distributed:
      child_runner.alg.broadcast_parameters()
    child_runner.logger.init_logging_writer()

    promotion_reason: str | None = None
    failure_reason: str | None = None
    metrics: dict[str, Any] = {}
    stage_status = "completed"
    is_primary = self._is_primary_process(child_runner)

    try:
      while child_runner.current_learning_iteration < stage_budget:
        iteration_index = child_runner.current_learning_iteration
        start = time.time()
        with torch.inference_mode():
          for _ in range(child_runner.cfg["num_steps_per_env"]):
            actions = child_runner.alg.act(obs)
            obs, rewards, dones, extras = child_runner.env.step(
              actions.to(child_runner.env.device)
            )
            if child_runner.cfg.get("check_for_nan", True):
              check_nan(obs, rewards, dones)
            obs = obs.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            child_runner.alg.process_env_step(obs, rewards, dones, extras)
            intrinsic_rewards = (
              child_runner.alg.intrinsic_rewards
              if child_runner.cfg["algorithm"]["rnd_cfg"]
              else None
            )
            child_runner.logger.process_env_step(
              rewards, dones, extras, intrinsic_rewards
            )
            metrics_manager = child_runner.env.unwrapped.metrics_manager
            if getattr(metrics_manager, "active_terms", None):
              monitor.observe_step(
                metrics_manager.active_terms, metrics_manager._step_values
              )
          collect_time = time.time() - start
          start = time.time()
          child_runner.alg.compute_returns(obs)

        loss_dict = child_runner.alg.update()
        learn_time = time.time() - start
        monitor.finish_iteration(iteration_index + 1)
        child_runner.logger.log(
          it=iteration_index,
          start_it=stage_start_iteration,
          total_it=stage_budget,
          collect_time=collect_time,
          learn_time=learn_time,
          loss_dict=loss_dict,
          learning_rate=child_runner.alg.learning_rate,
          action_std=child_runner.alg.get_policy().output_std,
          rnd_weight=(
            child_runner.alg.rnd.weight
            if child_runner.cfg["algorithm"]["rnd_cfg"]
            else None
          ),
        )

        child_runner.current_learning_iteration = iteration_index + 1
        self.current_learning_iteration += 1
        reduced_totals = self._reduce_monitor_totals(
          child_runner, monitor.aggregate_totals()
        )
        metrics = metrics_from_totals(reduced_totals)

        if self._should_save_checkpoint(child_runner.current_learning_iteration):
          latest_checkpoint = self._save_stage_checkpoint(
            child_runner=child_runner,
            stage_dir=stage_dir,
          )
          self._update_stage_record(
            stage_index=stage_index,
            status="in_progress",
            load_mode=load_mode,
            latest_checkpoint=latest_checkpoint,
            stage_iteration=child_runner.current_learning_iteration,
            promotion_reason=None,
            metrics=metrics,
            is_primary=is_primary,
          )
          self._save_root_checkpoint(child_runner)
          self._distributed_barrier(child_runner)

        decision = evaluate_promotion(
          stage_name,
          self.curriculum_cfg,
          child_runner.current_learning_iteration,
          metrics,
        )
        if decision.stop:
          stage_status = "failed"
          promotion_reason = decision.reason
          failure_reason = decision.reason
          metrics = decision.metrics or metrics
          break
        if decision.promote:
          promotion_reason = decision.reason or "threshold_met"
          metrics = decision.metrics or metrics
          break

      if promotion_reason is None and stage_status != "failed":
        promotion_reason = "max_iterations_reached"
    except Exception as exc:  # noqa: BLE001
      stage_status = "failed"
      failure_reason = str(exc)
      metrics = monitor.aggregate_metrics()
      self._update_stage_record(
        stage_index=stage_index,
        status="failed",
        load_mode=load_mode,
        latest_checkpoint=latest_checkpoint,
        stage_iteration=child_runner.current_learning_iteration,
        promotion_reason="failed_health_check",
        metrics=metrics,
        failure_reason=failure_reason,
        is_primary=is_primary,
      )
      self._save_root_checkpoint(child_runner)
      raise
    finally:
      child_runner.logger.stop_logging_writer()

    latest_checkpoint = self._save_stage_checkpoint(
      child_runner=child_runner, stage_dir=stage_dir
    )
    self._copy_latest_policy(stage_dir, child_runner=child_runner)
    self._update_stage_record(
      stage_index=stage_index,
      status=stage_status,
      load_mode=load_mode,
      latest_checkpoint=latest_checkpoint,
      stage_iteration=child_runner.current_learning_iteration,
      promotion_reason=promotion_reason,
      metrics=metrics,
      failure_reason=failure_reason,
      is_primary=is_primary,
    )
    self._save_root_checkpoint(child_runner)
    self._distributed_barrier(child_runner)

    return {
      "status": stage_status,
      "latest_checkpoint": latest_checkpoint,
      "failure_reason": failure_reason,
      "promotion_reason": promotion_reason,
    }

  def _save_stage_checkpoint(
    self,
    *,
    child_runner: VelocityOnPolicyRunner,
    stage_dir: Path,
  ) -> Path:
    checkpoint_path = stage_dir / f"model_{child_runner.current_learning_iteration}.pt"
    if self._is_primary_process(child_runner):
      child_runner.save(str(checkpoint_path))
    return checkpoint_path

  def _should_save_checkpoint(self, stage_iteration: int) -> bool:
    save_interval = int(self.cfg.get("save_interval", 0) or 0)
    return save_interval > 0 and stage_iteration % save_interval == 0

  def _save_root_checkpoint(self, child_runner: VelocityOnPolicyRunner) -> None:
    if self.log_dir is None or not self._is_primary_process(child_runner):
      return
    checkpoint_path = self.log_dir / f"model_{self.current_learning_iteration}.pt"
    self.save(str(checkpoint_path))

  def _record_stage_source(
    self,
    stage_index: int,
    checkpoint: Path | None,
    load_mode: str,
    *,
    is_primary: bool,
  ) -> None:
    manifest = manifest_for_update(self._manifest)
    stage_record = manifest["stages"][stage_index]
    stage_record["source_checkpoint"] = self._relative_to_run(checkpoint)
    stage_record["load_mode"] = load_mode
    manifest["updated_at"] = self._now()
    self._manifest = manifest
    if is_primary:
      write_manifest(self._manifest_path, self._manifest)

  def _update_stage_record(
    self,
    *,
    stage_index: int,
    status: str,
    load_mode: str,
    latest_checkpoint: Path | None,
    stage_iteration: int,
    promotion_reason: str | None,
    metrics: dict[str, Any],
    failure_reason: str | None = None,
    is_primary: bool,
  ) -> None:
    manifest = manifest_for_update(self._manifest)
    stage_record = manifest["stages"][stage_index]
    stage_record["status"] = status
    stage_record["load_mode"] = load_mode
    stage_record["latest_checkpoint"] = self._relative_to_run(latest_checkpoint)
    stage_record["stage_iteration"] = stage_iteration
    stage_record["global_iteration"] = self.current_learning_iteration
    stage_record["promotion_reason"] = promotion_reason
    stage_record["metrics"] = metrics
    if failure_reason is not None:
      stage_record["failure_reason"] = failure_reason
      manifest["failure_reason"] = failure_reason
      manifest["status"] = "failed"
    if status == "completed":
      stage_record["completed_at"] = self._now()
      if stage_index not in manifest["completed_stage_indices"]:
        manifest["completed_stage_indices"].append(stage_index)
      manifest["active_stage_index"] = stage_index + 1
      manifest["latest_checkpoint"] = self._relative_to_run(latest_checkpoint)
    elif status == "in_progress":
      manifest["active_stage_index"] = stage_index
      manifest["latest_checkpoint"] = self._relative_to_run(latest_checkpoint)
    else:
      manifest["active_stage_index"] = stage_index
      manifest["latest_checkpoint"] = self._relative_to_run(latest_checkpoint)
    manifest["updated_at"] = self._now()
    self._manifest = manifest
    if is_primary:
      write_manifest(self._manifest_path, self._manifest)

  def _stage_log_dir(self, stage_index: int, stage_name: str) -> Path:
    assert self.log_dir is not None
    return self.log_dir / "stages" / f"{stage_index:02d}_{stage_name}"

  def _copy_latest_policy(
    self,
    stage_dir: Path,
    *,
    child_runner: VelocityOnPolicyRunner,
  ) -> None:
    if self.log_dir is None or not self._is_primary_process(child_runner):
      return
    policy_path = stage_dir / "policy.onnx"
    if policy_path.exists():
      shutil.copy2(policy_path, self.log_dir / "policy.onnx")

  def _relative_to_run(self, path: Path | None) -> str | None:
    if path is None or self.log_dir is None:
      return None if path is None else str(path)
    try:
      return str(path.relative_to(self.log_dir))
    except ValueError:
      return str(path)

  def _resolve_run_path(self, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() or self.log_dir is None:
      return path
    return self.log_dir / path

  def _resolve_optional_run_path(self, path_str: str | None) -> Path | None:
    if path_str is None:
      return None
    return self._resolve_run_path(path_str)

  def _close_child_env(self, stage_env: RslRlVecEnvWrapper) -> None:
    stage_env.close()
    if stage_env is self.env:
      self._bootstrap_env_released = True
      stage_env.close = lambda: None  # type: ignore[method-assign]

  def _release_bootstrap_env(self) -> None:
    if self._bootstrap_env_released:
      return
    self._close_child_env(self.env)

  def _stage_status(self, stage_index: int) -> str:
    return str(self._manifest["stages"][stage_index].get("status") or "pending")

  @staticmethod
  def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
