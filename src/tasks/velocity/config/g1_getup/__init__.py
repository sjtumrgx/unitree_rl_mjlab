"""Task registration for Unitree G1 HoST get-up."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_g1_getup_env_cfg
from .rl_cfg import unitree_g1_getup_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-G1-GetUp",
  env_cfg=unitree_g1_getup_env_cfg(terrain="ground"),
  play_env_cfg=unitree_g1_getup_env_cfg(terrain="ground", play=True),
  rl_cfg=unitree_g1_getup_ppo_runner_cfg(terrain="ground"),
  runner_cls=VelocityOnPolicyRunner,
)
