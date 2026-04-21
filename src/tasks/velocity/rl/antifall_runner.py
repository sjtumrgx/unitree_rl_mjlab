"""Runner wrapper that exports deploy.yaml for anti-fall policies."""

from __future__ import annotations

import os

from .antifall_deploy_contract import write_antifall_deploy_yaml
from .runner import VelocityOnPolicyRunner


class AntiFallOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    write_antifall_deploy_yaml(
      self.env.unwrapped, os.path.join(policy_path, "params", "deploy.yaml")
    )
