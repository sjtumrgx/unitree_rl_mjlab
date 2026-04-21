import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _should_upload_model_artifacts(self) -> bool:
    return bool(self.cfg.get("upload_model", True))

  def save(self, path: str, infos=None):
    original_save_model = self.logger.save_model
    if not self._should_upload_model_artifacts():
      self.logger.save_model = lambda *args, **kwargs: None
    try:
      super().save(path, infos)
    finally:
      self.logger.save_model = original_save_model

    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"] and self._should_upload_model_artifacts():
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
