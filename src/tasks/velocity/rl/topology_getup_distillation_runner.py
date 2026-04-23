"""Distillation runner for topology get-up tasks."""

from __future__ import annotations

import os

import torch
import wandb
from rsl_rl.runners.distillation_runner import DistillationRunner

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

from src.tasks.velocity.rl.topology_getup_analysis import export_topology_analysis_to_onnx
from src.tasks.velocity.rl.topology_getup_artifacts import write_topology_getup_artifact_manifest
from src.tasks.velocity.rl.topology_getup_contract import (
  get_support_geometry_metadata,
  write_topology_getup_deploy_yaml,
)


class TopologyGetupDistillationRunner(DistillationRunner):
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    self._teacher_load_path = train_cfg.get("teacher_load_path", "")
    self._upload_model = bool(train_cfg.get("upload_model", True))
    super().__init__(env, train_cfg, log_dir, device)

  def _maybe_load_teacher(self) -> None:
    if self.alg.teacher_loaded:
      return
    if not self._teacher_load_path:
      raise ValueError(
        "Topology get-up distillation requires 'teacher_load_path' pointing to a PPO teacher checkpoint."
      )
    self.load(self._teacher_load_path, load_cfg={"teacher": True, "iteration": False}, strict=False)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    self._maybe_load_teacher()
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None) -> None:
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self._upload_model:
      self.logger.save_model(path, self.current_learning_iteration)

    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    analysis_path = export_topology_analysis_to_onnx(self.alg.get_policy(), policy_path)
    logger_type = getattr(self.logger, "logger_type", self.cfg.get("logger", "tensorboard")).lower()
    run_name: str = (
      wandb.run.name if logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    metadata.update(get_support_geometry_metadata(self.env.unwrapped))
    if analysis_path is not None:
      metadata["topology_analysis_export"] = os.path.basename(analysis_path)
    metadata["distillation_mode"] = "teacher_student_topology_bottleneck"
    attach_metadata_to_onnx(onnx_path, metadata)
    write_topology_getup_deploy_yaml(
      self.env.unwrapped,
      os.path.join(policy_path, "params", "deploy.yaml"),
    )
    write_topology_getup_artifact_manifest(
      output_dir=policy_path,
      experiment_name=self.cfg.get("experiment_name", ""),
      checkpoint_path=path,
      support_geometry_interface_version=metadata["support_geometry_interface_version"],
      policy_analysis_path=analysis_path,
      distillation_mode=metadata["distillation_mode"],
      teacher_checkpoint=self._teacher_load_path,
    )
    if logger_type in ["wandb"] and self._upload_model:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
