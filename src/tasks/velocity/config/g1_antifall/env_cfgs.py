"""Stage-configured Unitree G1 anti-fall velocity environment scaffolds."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity import mdp
from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
)

_ALLOWED_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)
_RECOVERY_WINDOW_S = 2.0
_TRACKING_THRESHOLD = 0.35
_YAW_THRESHOLD = 0.35
_TILT_THRESHOLD = 0.30
_STAGE2_HARD_POSE_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (0.0, 0.05),
  "roll": (-0.35, 0.35),
  "pitch": (-0.35, 0.35),
  "yaw": (-3.14, 3.14),
}
_STAGE2_HARD_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.25, 0.25),
  "roll": (-0.5, 0.5),
  "pitch": (-0.5, 0.5),
  "yaw": (-0.5, 0.5),
}


def _apply_antifall_actor_contract(
  cfg: ManagerBasedRlEnvCfg,
  *,
  history_length: int = 3,
) -> None:
  actor_obs = cfg.observations["actor"]
  actor_obs.terms = {
    name: term
    for name, term in actor_obs.terms.items()
    if name in _ALLOWED_ACTOR_TERMS
  }
  actor_obs.history_length = history_length


def _apply_antifall_critic_context(cfg: ManagerBasedRlEnvCfg) -> None:
  critic_obs = cfg.observations["critic"]
  critic_obs.terms["disturbance_metadata"] = ObservationTermCfg(
    func=mdp.disturbance_metadata,
    params={
      "window_s": _RECOVERY_WINDOW_S,
    },
  )
  critic_obs.terms["recovery_features"] = ObservationTermCfg(
    func=mdp.recovery_features,
    params={
      "command_name": "twist",
      "window_s": _RECOVERY_WINDOW_S,
      "tracking_threshold": _TRACKING_THRESHOLD,
      "yaw_threshold": _YAW_THRESHOLD,
      "tilt_threshold": _TILT_THRESHOLD,
    },
  )


def _apply_antifall_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards["upright_recoverability"] = RewardTermCfg(
    func=mdp.upright_recoverability,
    weight=0.2,
  )
  cfg.rewards["recovery_quality"] = RewardTermCfg(
    func=mdp.recovery_quality,
    weight=0.5,
    params={
      "command_name": "twist",
      "window_s": _RECOVERY_WINDOW_S,
    },
  )
  cfg.rewards["standing_stability"] = RewardTermCfg(
    func=mdp.standing_stability,
    weight=0.2,
    params={
      "command_name": "twist",
    },
  )
  cfg.rewards["recovery_completion_bonus"] = RewardTermCfg(
    func=mdp.recovery_completion_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      "window_s": _RECOVERY_WINDOW_S,
      "tracking_threshold": _TRACKING_THRESHOLD,
      "yaw_threshold": _YAW_THRESHOLD,
      "tilt_threshold": _TILT_THRESHOLD,
    },
  )


def _apply_antifall_metrics(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.metrics["disturbance_window_active"] = MetricsTermCfg(
    func=mdp.disturbance_window_active,
    params={"window_s": _RECOVERY_WINDOW_S},
  )
  cfg.metrics["disturbance_magnitude"] = MetricsTermCfg(
    func=mdp.disturbance_magnitude,
  )
  cfg.metrics["controllable_locomotion"] = MetricsTermCfg(
    func=mdp.controllable_locomotion,
    params={
      "command_name": "twist",
      "tracking_threshold": _TRACKING_THRESHOLD,
      "yaw_threshold": _YAW_THRESHOLD,
      "tilt_threshold": _TILT_THRESHOLD,
    },
  )
  cfg.metrics["disturbance_count"] = MetricsTermCfg(
    func=mdp.disturbance_count,
  )
  cfg.metrics["recovery_success_count"] = MetricsTermCfg(
    func=mdp.recovery_success_count,
    params={
      "command_name": "twist",
      "window_s": _RECOVERY_WINDOW_S,
      "tracking_threshold": _TRACKING_THRESHOLD,
      "yaw_threshold": _YAW_THRESHOLD,
      "tilt_threshold": _TILT_THRESHOLD,
    },
  )
  cfg.metrics["recovery_latency"] = MetricsTermCfg(
    func=mdp.recovery_latency,
    params={
      "command_name": "twist",
      "window_s": _RECOVERY_WINDOW_S,
      "tracking_threshold": _TRACKING_THRESHOLD,
      "yaw_threshold": _YAW_THRESHOLD,
      "tilt_threshold": _TILT_THRESHOLD,
    },
  )


def _tune_command_ranges(
  cfg: ManagerBasedRlEnvCfg,
  *,
  rel_standing_envs: float,
  lin_vel_x: tuple[float, float],
  lin_vel_y: tuple[float, float],
  ang_vel_z: tuple[float, float],
) -> None:
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = rel_standing_envs
  twist_cmd.ranges.lin_vel_x = lin_vel_x
  twist_cmd.ranges.lin_vel_y = lin_vel_y
  twist_cmd.ranges.ang_vel_z = ang_vel_z


def _configure_push_profile(
  cfg: ManagerBasedRlEnvCfg,
  *,
  interval_range_s: tuple[float, float],
  velocity_range: dict[str, tuple[float, float]],
) -> None:
  push_event = cfg.events.get("push_robot")
  if push_event is None:
    return
  push_event.interval_range_s = interval_range_s
  push_event.params["velocity_range"] = dict(velocity_range)


def _current_reset_ranges(
  cfg: ManagerBasedRlEnvCfg,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
  params = cfg.events["reset_base"].params
  if "nominal_pose_range" in params:
    pose_range = dict(params["nominal_pose_range"])
    velocity_range = dict(params.get("nominal_velocity_range") or {})
  else:
    pose_range = dict(params["pose_range"])
    velocity_range = dict(params.get("velocity_range") or {})
  return pose_range, velocity_range


def _configure_antifall_reset(
  cfg: ManagerBasedRlEnvCfg,
  *,
  hard_reset_prob: float = 0.0,
  hard_pose_range: dict[str, tuple[float, float]] | None = None,
  hard_velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
  nominal_pose_range, nominal_velocity_range = _current_reset_ranges(cfg)
  cfg.events["reset_base"].func = mdp.reset_root_state_mixed
  cfg.events["reset_base"].params = {
    "nominal_pose_range": nominal_pose_range,
    "nominal_velocity_range": nominal_velocity_range,
    "hard_pose_range": hard_pose_range,
    "hard_velocity_range": hard_velocity_range,
    "hard_reset_prob": hard_reset_prob,
  }


def _configure_antifall_push_tracking(cfg: ManagerBasedRlEnvCfg) -> None:
  push_event = cfg.events.get("push_robot")
  if push_event is None:
    return
  push_event.func = mdp.push_by_setting_velocity_with_history


def _apply_antifall_helpers(
  cfg: ManagerBasedRlEnvCfg,
  *,
  hard_reset_prob: float = 0.0,
  hard_pose_range: dict[str, tuple[float, float]] | None = None,
  hard_velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
  _apply_antifall_critic_context(cfg)
  _apply_antifall_rewards(cfg)
  _apply_antifall_metrics(cfg)
  _configure_antifall_reset(
    cfg,
    hard_reset_prob=hard_reset_prob,
    hard_pose_range=hard_pose_range,
    hard_velocity_range=hard_velocity_range,
  )
  _configure_antifall_push_tracking(cfg)


def _make_antifall_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  _apply_antifall_actor_contract(cfg)
  cfg.episode_length_s = 25.0
  if play:
    cfg.sim.nconmax = 256
  return cfg


def unitree_g1_antifall_stage0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 0 stable flat-ground anti-fall scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  cfg.events.pop("push_robot", None)
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.2,
    lin_vel_x=(-0.4, 1.0),
    lin_vel_y=(-0.3, 0.3),
    ang_vel_z=(-0.6, 0.6),
  )
  _apply_antifall_helpers(cfg)
  return cfg


def unitree_g1_antifall_stage1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 1 flat push-recovery scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.15,
    lin_vel_x=(-0.5, 1.25),
    lin_vel_y=(-0.4, 0.4),
    ang_vel_z=(-0.8, 0.8),
  )
  if not play:
    _configure_push_profile(
      cfg,
      interval_range_s=(4.0, 6.0),
      velocity_range={
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    )
  _apply_antifall_helpers(cfg)
  return cfg


def unitree_g1_antifall_stage2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 2 harder flat recovery scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.1,
    lin_vel_x=(-0.75, 1.5),
    lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-1.0, 1.0),
  )
  if not play:
    _configure_push_profile(
      cfg,
      interval_range_s=(3.0, 5.0),
      velocity_range={
        "x": (-0.75, 0.75),
        "y": (-0.75, 0.75),
        "z": (-0.5, 0.5),
        "roll": (-0.7, 0.7),
        "pitch": (-0.7, 0.7),
        "yaw": (-1.0, 1.0),
      },
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)
  _apply_antifall_helpers(
    cfg,
    hard_reset_prob=0.2,
    hard_pose_range=_STAGE2_HARD_POSE_RANGE,
    hard_velocity_range=_STAGE2_HARD_VELOCITY_RANGE,
  )
  return cfg


def unitree_g1_antifall_stage3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 3 walking-biased flat push-kick recovery scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.08,
    lin_vel_x=(-0.9, 1.6),
    lin_vel_y=(-0.6, 0.6),
    ang_vel_z=(-1.1, 1.1),
  )
  if not play:
    _configure_push_profile(
      cfg,
      interval_range_s=(2.75, 4.25),
      velocity_range={
        "x": (-0.9, 0.9),
        "y": (-0.85, 0.85),
        "z": (-0.55, 0.55),
        "roll": (-0.8, 0.8),
        "pitch": (-0.8, 0.8),
        "yaw": (-1.05, 1.05),
      },
    )
  _apply_antifall_helpers(
    cfg,
    hard_reset_prob=0.25,
    hard_pose_range=_STAGE2_HARD_POSE_RANGE,
    hard_velocity_range=_STAGE2_HARD_VELOCITY_RANGE,
  )
  return cfg


def unitree_g1_antifall_stage4a_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 4a lateral / asymmetric push-kick recovery scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.06,
    lin_vel_x=(-1.0, 1.75),
    lin_vel_y=(-0.7, 0.7),
    ang_vel_z=(-1.2, 1.2),
  )
  if not play:
    _configure_push_profile(
      cfg,
      interval_range_s=(2.25, 3.75),
      velocity_range={
        "x": (-0.75, 0.75),
        "y": (-1.15, 1.15),
        "z": (-0.65, 0.65),
        "roll": (-0.95, 0.95),
        "pitch": (-0.75, 0.75),
        "yaw": (-1.15, 1.15),
      },
    )
  _apply_antifall_helpers(
    cfg,
    hard_reset_prob=0.3,
    hard_pose_range=_STAGE2_HARD_POSE_RANGE,
    hard_velocity_range=_STAGE2_HARD_VELOCITY_RANGE,
  )
  return cfg


def unitree_g1_antifall_stage4b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 4b hardest mixed push-kick recovery scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.05,
    lin_vel_x=(-1.1, 1.9),
    lin_vel_y=(-0.8, 0.8),
    ang_vel_z=(-1.25, 1.25),
  )
  if not play:
    _configure_push_profile(
      cfg,
      interval_range_s=(1.75, 3.25),
      velocity_range={
        "x": (-1.25, 1.25),
        "y": (-1.25, 1.25),
        "z": (-0.75, 0.75),
        "roll": (-1.05, 1.05),
        "pitch": (-1.05, 1.05),
        "yaw": (-1.3, 1.3),
      },
    )
  _apply_antifall_helpers(
    cfg,
    hard_reset_prob=0.35,
    hard_pose_range=_STAGE2_HARD_POSE_RANGE,
    hard_velocity_range=_STAGE2_HARD_VELOCITY_RANGE,
  )
  return cfg


def unitree_g1_antifall_benchmark_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the deterministic anti-fall benchmark scaffold."""
  cfg = _make_antifall_flat_env_cfg(play=play)
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  cfg.observations["actor"].enable_corruption = False
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (0.0, 0.0),
    1: (0.0, 0.0),
    2: (0.0, 0.0),
  }
  _apply_antifall_helpers(cfg)
  return cfg
