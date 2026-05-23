"""Stage-configured Unitree G1 anti-fall velocity environment scaffolds."""

from collections import OrderedDict
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity import mdp
from src.tasks.velocity.config.g1.env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from src.tasks.velocity.mdp.getup.actions import RecoveryHybridJointPositionActionCfg

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
_ANTIFALL_GETUP_HARD_RESET_SCHEDULE = (
  # Expose recoveries early instead of waiting until the policy has already
  # specialized on walking-only tracking.  A tiny hard-reset rate from the
  # start gives PPO a stable stream of fallen-start episodes while the warm
  # start is still intact, then ramps as the policy becomes ready for more
  # recovery burden.
  {"step": 0, "prob": 0.02},
  {"step": 300, "prob": 0.05},
  {"step": 900, "prob": 0.10},
  {"step": 1800, "prob": 0.15},
)
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
  recovery_window_s: float | None = None,
  active: bool | None = None,
) -> None:
  push_event = cfg.events.get("push_robot")
  if push_event is None:
    return
  push_event.interval_range_s = interval_range_s
  push_event.params["velocity_range"] = dict(velocity_range)
  if recovery_window_s is not None:
    push_event.params["recovery_window_s"] = float(recovery_window_s)
  if active is not None:
    push_event.params["active"] = bool(active)


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
  hard_reset_prob_schedule: tuple[dict[str, float], ...] | None = None,
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
    "hard_reset_prob_schedule": hard_reset_prob_schedule,
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
  hard_reset_prob_schedule: tuple[dict[str, float], ...] | None = None,
  hard_pose_range: dict[str, tuple[float, float]] | None = None,
  hard_velocity_range: dict[str, tuple[float, float]] | None = None,
) -> None:
  _apply_antifall_critic_context(cfg)
  _apply_antifall_rewards(cfg)
  _apply_antifall_metrics(cfg)
  _configure_antifall_reset(
    cfg,
    hard_reset_prob=hard_reset_prob,
    hard_reset_prob_schedule=hard_reset_prob_schedule,
    hard_pose_range=hard_pose_range,
    hard_velocity_range=hard_velocity_range,
  )
  _configure_antifall_push_tracking(cfg)


def _move_randomize_terrain_before_root_reset(cfg: ManagerBasedRlEnvCfg) -> None:
  randomize = cfg.events.get("randomize_terrain")
  if randomize is None or getattr(randomize, "mode", None) != "reset":
    return

  ordered = OrderedDict()
  ordered["randomize_terrain"] = randomize
  for name, term in cfg.events.items():
    if name != "randomize_terrain":
      ordered[name] = term
  cfg.events = ordered


def _restore_antifall_getup_actor_contract(
  cfg: ManagerBasedRlEnvCfg,
  *,
  history_length: int = 6,
) -> None:
  """Keep AntiFall-GetUp recovery observations aligned with standalone GetUp.

  The ordinary AntiFall stages intentionally use a compact 3-frame proprio actor
  contract.  AntiFall-GetUp, however, is bootstrapped from the proven GetUp
  recovery policy; dropping its six-frame history and terrain scan destroys that
  recovery prior before PPO can combine it with walking.  Keep this richer
  contract local to the GetUp hybrid task so the Stage4b walking tasks remain
  unchanged.
  """

  actor_obs = cfg.observations["actor"]
  actor_obs.history_length = None
  for term_name in (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "getup_progress",
  ):
    term = actor_obs.terms.get(term_name)
    if term is not None:
      term.history_length = history_length

  bfm_term = actor_obs.terms.get("bfm_local_body_state")
  if bfm_term is not None:
    bfm_term.history_length = 0

  if "height_scan" not in actor_obs.terms:
    rough_ref = unitree_g1_rough_env_cfg(play=False)
    if not any(sensor.name == "terrain_scan" for sensor in (cfg.scene.sensors or ())):
      terrain_scan = next(
        sensor for sensor in (rough_ref.scene.sensors or ()) if sensor.name == "terrain_scan"
      )
      cfg.scene.sensors = (cfg.scene.sensors or ()) + (deepcopy(terrain_scan),)
    actor_obs.terms["height_scan"] = deepcopy(
      rough_ref.observations["actor"].terms["height_scan"]
    )
  actor_obs.terms["height_scan"].history_length = history_length


def _add_antifall_getup_recovery_phase_observations(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  term = ObservationTermCfg(
    func=mdp.host_getup_recovery_phase,
    history_length=0,
  )
  actor_terms = OrderedDict()
  inserted = False
  for name, existing_term in cfg.observations["actor"].terms.items():
    actor_terms[name] = existing_term
    if name == "getup_progress":
      actor_terms["recovery_phase"] = term
      inserted = True
  if not inserted:
    actor_terms["recovery_phase"] = term
  cfg.observations["actor"].terms = actor_terms
  cfg.observations["critic"].terms["recovery_phase"] = term


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
        "x": (-1.75, 1.75),
        "y": (-1.75, 1.75),
        "z": (-1, 1),
        "roll": (-1.25, 1.25),
        "pitch": (-1.25, 1.25),
        "yaw": (-1.5, 1.5),
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


def _add_antifall_rewards_after_getup_stack(cfg: ManagerBasedRlEnvCfg, source_rewards: dict) -> None:
  """Re-add locomotion and recovery terms after the GetUp stack trims rewards."""

  for reward_name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "body_orientation_l2",
    "pose",
  ):
    if reward_name in source_rewards:
      cfg.rewards[reward_name] = source_rewards[reward_name]
  _apply_antifall_rewards(cfg)


def _gate_getup_rewards_to_recovery_phase(cfg: ManagerBasedRlEnvCfg) -> None:
  """Prevent dense GetUp shaping from altering nominal warm-start walking."""

  getup_reward_names = (
    "host_task_reward",
    "host_lift_progress",
    "host_upright_progress",
    "host_support_relief",
    "host_action_smoothness",
    "host_joint_tracking",
    "host_style_pose",
    "host_feet_support",
    "host_hand_support_progress",
    "host_hand_push",
    "host_hand_contact_after_stand",
    "host_foot_contact_spread",
    "host_foot_flat",
    "host_foot_heading",
    "host_natural_stand_pose",
    "host_foot_orientation_penalty",
    "host_ankle_deviation_penalty",
    "host_target_standing",
    "getup_completion_bonus",
  )
  for reward_name in getup_reward_names:
    term = cfg.rewards.get(reward_name)
    if term is None or term.func is mdp.recovery_phase_reward:
      continue
    original_func = term.func
    original_params = dict(term.params)
    term.func = mdp.recovery_phase_reward
    term.params = {
      "reward_func": original_func,
      "fallen_height_threshold": 0.35,
      "fallen_tilt_threshold": 0.75,
      "window_s": 0.0,
      "include_disturbance_window": False,
      "include_near_failure_reset_window": True,
      **original_params,
    }


def unitree_g1_antifall_getup_env_cfg(
  play: bool = False,
  *,
  hard_reset_prob: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create walking anti-fall plus fallen GetUp recovery scaffold.

  This task intentionally combines the late anti-fall walking/push ladder with
  the repaired HoST-style GetUp action/reset/reward contract.  It trains one
  actor to track velocity commands, absorb disturbances, recover from hard
  fallen resets, and resume controllable locomotion.

  ``hard_reset_prob`` is exposed so diagnostics can enforce the BFM-style
  lifecycle order: start from nominal walking, then evaluate recovery after an
  explicit disturbance.  The default training contract now injects a small
  hard-reset rate from the beginning so the warm-started walking actor sees
  fallen recovery early instead of only after it has over-specialized.
  """

  from src.tasks.velocity.config.g1_getup.env_cfgs import (
    GETUP_FALLEN_ROOT_PRESETS,
    GETUP_SUCCESS_TORSO_HEIGHT,
    _GETUP_HARD_POSE_RANGE,
    _GETUP_HARD_VELOCITY_RANGE,
    _HOST_GETUP_INITIAL_ACTION_SCALE,
    _HOST_GETUP_MAX_ACTION_DELTA,
    _HOST_GETUP_MIN_ACTION_SCALE,
    _HOST_GETUP_UNACTUATED_TIMESTEPS,
    _GETUP_TRAIN_PRESET_WEIGHT_STAGES,
    _add_getup_stall_guard,
    _add_support_body_contact_sensor,
    _add_support_depth_camera,
    _apply_getup_nan_safety,
    _apply_host_effective_action_observations,
    _apply_host_getup_reward_stack,
    _apply_host_getup_safe_regularizers,
    _host_getup_stable_success_params,
  )

  cfg = unitree_g1_antifall_stage4b_env_cfg(play=play)
  _move_randomize_terrain_before_root_reset(cfg)
  locomotion_reward_source = dict(cfg.rewards)
  if not play:
    cfg.scene.num_envs = 4096
  cfg.episode_length_s = 30.0
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.host_unactuated_timesteps = _HOST_GETUP_UNACTUATED_TIMESTEPS  # type: ignore[attr-defined]
  cfg.host_reward_groups = ("task", "regu", "style", "target", "antifall")  # type: ignore[attr-defined]
  stage4b_action = unitree_g1_antifall_stage4b_env_cfg(play=True).actions["joint_pos"]
  cfg.actions["joint_pos"] = RecoveryHybridJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=stage4b_action.scale,
    use_default_offset=True,
    recovery_use_default_offset=False,
    recovery_window_s=_RECOVERY_WINDOW_S,
    fallen_height_threshold=0.35,
    fallen_tilt_threshold=0.75,
    # Exit the current-pose recovery action contract as soon as the coarse
    # torso/tilt gate says the robot is upright enough for the warm-started
    # walking prior.  Recovery commands stay quiet for the fixed BFM-style
    # disturbance window, then ramp briefly after this action phase exits.
    stable_upright_hold_steps=1,
    # Match BFM-Zero G1 fall-recovery control: actions are normalized/clipped
    # upstream, then converted to joint deltas with a 0.25 physical scale.
    recovery_action_scale=0.25,
    recovery_unactuated_timesteps=_HOST_GETUP_UNACTUATED_TIMESTEPS,
    walking_exit_max_delta=_HOST_GETUP_MAX_ACTION_DELTA,
    recovery_default_offset_joint_names=(
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ),
    max_delta=_HOST_GETUP_MAX_ACTION_DELTA,
  )
  _tune_command_ranges(
    cfg,
    rel_standing_envs=0.05,
    lin_vel_x=(-0.8, 1.6),
    lin_vel_y=(-0.5, 0.5),
    ang_vel_z=(-1.0, 1.0),
  )
  if "push_robot" not in cfg.events:
    cfg.events["push_robot"] = EventTermCfg(
      func=mdp.push_by_setting_velocity_with_history,
      mode="interval",
      params={"velocity_range": {}},
      interval_range_s=(2.0, 3.5),
    )
  if play:
    push_interval_range_s = (2.0, 3.5)
    push_velocity_range = {
      "x": (-1.4, 1.4),
      "y": (-1.4, 1.4),
      "z": (-0.9, 0.9),
      "roll": (-1.2, 1.2),
      "pitch": (-1.2, 1.2),
      "yaw": (-1.4, 1.4),
    }
  else:
    # Preserve the Stage4b walking actor before introducing full gate-strength
    # push/fall recovery.  The play diagnostic still uses the harder BFM-style
    # push profile; training starts from lower-frequency Stage2-scale pushes so
    # PPO does not immediately destroy command tracking.
    push_interval_range_s = (6.0, 8.0)
    push_velocity_range = {
      "x": (-0.75, 0.75),
      "y": (-0.75, 0.75),
      "z": (-0.5, 0.5),
      "roll": (-0.7, 0.7),
      "pitch": (-0.7, 0.7),
      "yaw": (-1.0, 1.0),
    }
  _configure_push_profile(
    cfg,
    interval_range_s=push_interval_range_s,
    velocity_range=push_velocity_range,
    recovery_window_s=_RECOVERY_WINDOW_S,
    active=True,
  )
  _add_support_depth_camera(cfg)
  _add_support_body_contact_sensor(cfg)
  _restore_antifall_getup_actor_contract(cfg)
  cfg.observations["actor"].terms["command"].func = mdp.recovery_phase_quiet_generated_commands
  cfg.observations["actor"].terms["command"].params["phase_attr"] = "_host_getup_recovery_phase_active"
  _add_antifall_getup_recovery_phase_observations(cfg)
  _add_getup_stall_guard(cfg)
  cfg.terminations["stalled_getup"].params["recovery_grace_s"] = _RECOVERY_WINDOW_S
  _apply_getup_nan_safety(cfg)
  _apply_host_effective_action_observations(cfg)
  scheduled_getup_hard_reset = hard_reset_prob is None and not play
  _apply_antifall_helpers(
    cfg,
    hard_reset_prob=0.0 if scheduled_getup_hard_reset else (0.15 if hard_reset_prob is None else float(hard_reset_prob)),
    hard_reset_prob_schedule=_ANTIFALL_GETUP_HARD_RESET_SCHEDULE if scheduled_getup_hard_reset else None,
    hard_pose_range=_GETUP_HARD_POSE_RANGE,
    hard_velocity_range=_GETUP_HARD_VELOCITY_RANGE,
  )
  cfg.events["reset_base"].func = mdp.reset_root_state_mixed_from_presets
  cfg.events["reset_base"].params.pop("hard_pose_range", None)
  cfg.events["reset_base"].params["presets"] = GETUP_FALLEN_ROOT_PRESETS
  cfg.events["reset_base"].params["preset_weight_stages"] = _GETUP_TRAIN_PRESET_WEIGHT_STAGES
  cfg.events["reset_base"].params["command_name"] = "twist"
  cfg.events["reset_base"].params["command_quiet_s"] = _RECOVERY_WINDOW_S
  cfg.events["ramp_recovery_exit_command"] = EventTermCfg(
    func=mdp.ramp_velocity_command_after_recovery_exit,
    mode="step",
    params={"command_name": "twist", "ramp_s": 1.0},
  )
  _apply_host_getup_safe_regularizers(cfg)
  _apply_host_getup_reward_stack(cfg)
  _add_antifall_rewards_after_getup_stack(cfg, locomotion_reward_source)
  cfg.rewards["recovery_quality"].params.update(
    {
      "require_fallen_or_near_failure": True,
      "fallen_height_threshold": 0.35,
      "fallen_tilt_threshold": 0.75,
    }
  )
  cfg.rewards["recovery_completion_bonus"].params.update(
    {
      "require_fallen_or_near_failure": True,
      "fallen_height_threshold": 0.35,
      "fallen_tilt_threshold": 0.75,
    }
  )
  cfg.rewards["post_recovery_resume_locomotion"] = RewardTermCfg(
    func=mdp.post_recovery_resume_locomotion,
    weight=1.0,
    params={
      "command_name": "twist",
      "recovery_window_s": _RECOVERY_WINDOW_S,
      "resume_window_s": 8.0,
      "fallen_height_threshold": 0.35,
      "fallen_tilt_threshold": 0.75,
    },
  )
  cfg.metrics["post_recovery_resume_locomotion"] = MetricsTermCfg(
    func=mdp.post_recovery_resume_locomotion,
    params={
      "command_name": "twist",
      "recovery_window_s": _RECOVERY_WINDOW_S,
      "resume_window_s": 8.0,
      "fallen_height_threshold": 0.35,
      "fallen_tilt_threshold": 0.75,
    },
  )
  _gate_getup_rewards_to_recovery_phase(cfg)
  if not play:
    cfg.events["getup_assist_force"] = EventTermCfg(
      func=mdp.apply_host_getup_assist_force,
      mode="step",
      params={
        "initial_force_n": 100.0,
        "initial_action_scale": _HOST_GETUP_INITIAL_ACTION_SCALE,
        "success_height_threshold": GETUP_SUCCESS_TORSO_HEIGHT,
        "force_decay_n": 20.0,
        "action_scale_decay": 0.02,
        "min_force_n": 0.0,
        "min_action_scale": _HOST_GETUP_MIN_ACTION_SCALE,
        "unactuated_timesteps": _HOST_GETUP_UNACTUATED_TIMESTEPS,
        "orientation_projected_gravity_z_max": -0.8,
        "no_orientation_gate": True,
        "stable_success_required": True,
        "upright_alignment_threshold": 0.85,
        **_host_getup_stable_success_params(),
        "taper_start_height": 0.35,
        "taper_end_height": GETUP_SUCCESS_TORSO_HEIGHT,
        "no_assist_probability_initial": 0.10,
        "no_assist_probability": 0.75,
        "no_assist_ramp_start_progress": 0.5,
        "no_assist_ramp_end_progress": 1.0,
        "recovery_phase_only": True,
        "fallen_height_threshold": 0.35,
        "fallen_tilt_threshold": 0.75,
        "recovery_window_s": _RECOVERY_WINDOW_S,
        "include_disturbance_window": False,
        "include_near_failure_reset_window": True,
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    )
    cfg.metrics["getup_assist_force_n"] = MetricsTermCfg(
      func=mdp.getup_assist_force_n,
      params={"initial_force_n": 100.0},
    )
    cfg.metrics["getup_action_rescale"] = MetricsTermCfg(
      func=mdp.getup_action_rescale,
      params={"initial_action_scale": _HOST_GETUP_INITIAL_ACTION_SCALE},
    )
    paired_forced_fall_interval_s = (5.0, 7.0)
    cfg.events["mid_episode_forced_fall"] = EventTermCfg(
      func=mdp.reset_paired_fallen_state_from_presets,
      mode="interval",
      interval_range_s=paired_forced_fall_interval_s,
      params={
        "presets": GETUP_FALLEN_ROOT_PRESETS,
        "preset_weight_stages": _GETUP_TRAIN_PRESET_WEIGHT_STAGES,
        "velocity_range": _GETUP_HARD_VELOCITY_RANGE,
        "joint_position_noise_range": (-0.05, 0.05),
        "joint_velocity_range": (-0.5, 0.5),
        "reset_actions": True,
        "command_name": "twist",
        "command_quiet_s": _RECOVERY_WINDOW_S,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )
  cfg.events["reset_robot_joints"].func = mdp.reset_joints_mixed_by_antifall_state
  cfg.events["reset_robot_joints"].params = {
    "nominal_position_range": (-0.0, 0.0),
    "nominal_velocity_range": (-0.0, 0.0),
    "preset_position_noise_range": (-0.05, 0.05),
    "preset_velocity_range": (-0.5, 0.5),
    "asset_cfg": SceneEntityCfg("robot"),
  }
  return cfg


def unitree_g1_antifall_getup_recovery_warmup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a fallen-start recovery warmup with the final AntiFall-GetUp actor contract.

  This is the missing bridge between the Stage4b walking actor and the full
  walking+fall+resume task: it keeps the exact final actor/action tensor shapes
  but removes walking commands and interval pushes so PPO first learns a floor
  recovery prior from paired fallen presets.
  """

  from src.tasks.velocity.config.g1_getup.env_cfgs import (
    GETUP_FALLEN_ROOT_PRESETS,
    _GETUP_HARD_VELOCITY_RANGE,
    _GETUP_TRAIN_PRESET_WEIGHT_STAGES,
    _apply_zero_command_profile,
  )

  cfg = unitree_g1_antifall_getup_env_cfg(play=play, hard_reset_prob=1.0)
  _apply_zero_command_profile(cfg)
  cfg.events.pop("push_robot", None)
  cfg.events.pop("mid_episode_forced_fall", None)
  cfg.events["reset_base"].func = mdp.reset_root_state_from_presets
  cfg.events["reset_base"].params = {
    "presets": GETUP_FALLEN_ROOT_PRESETS,
    "preset_weight_stages": _GETUP_TRAIN_PRESET_WEIGHT_STAGES,
    "velocity_range": _GETUP_HARD_VELOCITY_RANGE,
  }
  cfg.events["reset_robot_joints"].func = mdp.reset_joints_from_presets
  cfg.events["reset_robot_joints"].params = {
    "position_noise_range": (-0.05, 0.05),
    "velocity_range": (-0.5, 0.5),
    "asset_cfg": SceneEntityCfg("robot"),
  }
  cfg.terminations["stalled_getup"].params["recovery_grace_s"] = 0.0

  for reward_name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "standing_stability",
    "upright_recoverability",
    "recovery_quality",
    "recovery_completion_bonus",
    "post_recovery_resume_locomotion",
  ):
    cfg.rewards.pop(reward_name, None)

  for reward_name, term in list(cfg.rewards.items()):
    if term.func is mdp.recovery_phase_reward:
      term.func = term.params.pop("reward_func")
      for key in (
        "fallen_height_threshold",
        "fallen_tilt_threshold",
        "window_s",
        "include_disturbance_window",
        "include_near_failure_reset_window",
      ):
        term.params.pop(key, None)

  cfg.host_reward_groups = ("task", "regu", "style", "target")  # type: ignore[attr-defined]
  return cfg
