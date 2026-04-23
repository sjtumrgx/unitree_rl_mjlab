"""Custom runner for topology get-up tasks with SGI export metadata."""

from __future__ import annotations

import os

import wandb

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

from src.tasks.velocity.rl.runner import VelocityOnPolicyRunner
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.tasks.velocity.rl.topology_getup_analysis import export_topology_analysis_to_onnx
from src.tasks.velocity.rl.topology_getup_artifacts import write_topology_getup_artifact_manifest
from src.tasks.velocity.rl.topology_getup_contract import (
  get_support_geometry_metadata,
  write_topology_getup_deploy_yaml,
)


class TopologyGetupOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None):
    original_save_model = self.logger.save_model
    if not self._should_upload_model_artifacts():
      self.logger.save_model = lambda *args, **kwargs: None
    try:
      MjlabOnPolicyRunner.save(self, path, infos)
    finally:
      self.logger.save_model = original_save_model

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
    )
    if logger_type in ["wandb"] and self._should_upload_model_artifacts():
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
