"""Task registration for G1 parkour MuJoCo play."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  PARKOUR_TASK_ID,
  unitree_g1_parkour_env_cfg,
)
from .rl_cfg import unitree_g1_parkour_runner_cfg


register_mjlab_task(
  task_id=PARKOUR_TASK_ID,
  env_cfg=unitree_g1_parkour_env_cfg(),
  play_env_cfg=unitree_g1_parkour_env_cfg(play=True),
  rl_cfg=unitree_g1_parkour_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
