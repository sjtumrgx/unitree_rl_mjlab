"""Terrain get-up environment scaffolds for Unitree G1.

This family is intentionally isolated from the existing flat anti-fall tasks so the
new HoST get-up work does not change current training task behavior.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains.config import ALL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.velocity.mdp as mdp

from src.tasks.velocity.config.g1_23dof.env_cfgs import unitree_g1_23dof_rough_env_cfg
from src.tasks.velocity.mdp.getup.actions import HostRelativeJointPositionActionCfg
from src.tasks.velocity.config.g1_antifall.env_cfgs import (
  _apply_antifall_actor_contract,
  _apply_antifall_helpers,
)


GETUP_TRAIN_NUM_ENVS = 4096
GETUP_EPISODE_LENGTH_S = 12.0
GETUP_STALL_MIN_STEPS = 600
GETUP_SUCCESS_TORSO_HEIGHT = 0.55
HOST_SOURCE_TASKS = {
  "mixed": "g1_mixed_terrains",
  "ground": "g1_ground",
  "platform": "g1_platform",
  "wall": "g1_wall",
  "slope": "g1_slope",
}

def _getup_initial_assist_force(terrain: str) -> float:
  """Small bootstrap force; play-like no-assist episodes remain the main path."""

  return float(HOST_TERRAIN_PARITY[terrain]["pull_force_n"])


HOST_TERRAIN_PARITY = {
  "mixed": {
    "num_rows": 10,
    "num_cols": 20,
    "terrain_proportions": (0.25, 0.25, 0.25, 0.25),
    "mjlab_terrain_names": ("flat", "pyramid_stairs", "hf_pyramid_slope", "random_rough"),
    "curriculum": True,
    "max_init_terrain_level": 5,
    "target_base_height_phase1": 0.45,
    "target_base_height_phase2": 0.45,
    "target_base_height_phase3": 0.65,
    "pull_force_n": 120,
  },
  "ground": {
    "num_rows": 1,
    "num_cols": 20,
    "terrain_proportions": (1, 0.0, 0, 0, 0),
    "mjlab_terrain_names": ("flat",),
    "curriculum": True,
    "max_init_terrain_level": 5,
    "target_base_height_phase1": 0.45,
    "target_base_height_phase2": 0.45,
    "target_base_height_phase3": 0.65,
    # Keep the ground assist as a sparse bootstrap, not the primary solution.
    # The 200N/50% mix learned assisted get-up but stayed 0/64 in play-like
    # diagnostics through 1000 iterations; bias training toward no-assist
    # dynamics and let the wrench only seed occasional upright examples.
    "pull_force_n": 120,
  },
  "platform": {
    "num_rows": 8,
    "num_cols": 8,
    "terrain_proportions": (0, 0.0, 1, 0, 0),
    "mjlab_terrain_names": ("pyramid_stairs",),
    "curriculum": False,
    "max_init_terrain_level": 3,
    "target_base_height_phase1": 0.45,
    "target_base_height_phase2": 0.45,
    "target_base_height_phase3": 0.65,
    "pull_force_n": 100,
  },
  "wall": {
    "num_rows": 4,
    "num_cols": 5,
    "terrain_proportions": (1, 0.0, 0, 0, 0),
    "mjlab_terrain_names": ("hf_pyramid_slope",),
    "curriculum": False,
    "max_init_terrain_level": 3,
    "target_base_height_phase1": 0.45,
    "target_base_height_phase2": 0.45,
    "target_base_height_phase3": 0.65,
    "pull_force_n": 100,
  },
  "slope": {
    "num_rows": 4,
    "num_cols": 8,
    "terrain_proportions": (1, 0, 0, 0, 0),
    "mjlab_terrain_names": ("hf_pyramid_slope",),
    "curriculum": False,
    "max_init_terrain_level": 3,
    "target_base_height_phase1": 0.4,
    "target_base_height_phase2": 0.4,
    "target_base_height_phase3": 0.6,
    "pull_force_n": 100,
  },
}

_GETUP_HARD_POSE_RANGE = {
  "x": (-0.3, 0.3),
  "y": (-0.3, 0.3),
  "z": (0.0, 0.08),
  "roll": (-2.2, 2.2),
  "pitch": (-2.2, 2.2),
  "yaw": (-3.14, 3.14),
}
_GETUP_HARD_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.25, 0.25),
  "roll": (-0.75, 0.75),
  "pitch": (-0.75, 0.75),
  "yaw": (-0.75, 0.75),
}
_HOST_GETUP_UNACTUATED_TIMESTEPS = 30
_HOST_GETUP_INITIAL_ACTION_SCALE = 1.0
_HOST_GETUP_MIN_ACTION_SCALE = 0.25
_HOST_GETUP_MAX_ACTION_DELTA = 1.0
_GETUP_TRAIN_PRESET_WEIGHT_STAGES = (
  {"step": 0, "weights": (0.0, 0.25, 0.25, 0.5)},
  {"step": 48, "weights": (0.15, 0.25, 0.25, 0.35)},
  {"step": 120, "weights": (0.25, 0.25, 0.25, 0.25)},
)
_GETUP_PLAY_PRESET_WEIGHT_STAGES = (
  {"step": 0, "weights": (0.25, 0.25, 0.25, 0.25)},
)
_HOST_GETUP_TARGET_JOINT_ANGLES = {
  "left_hip_yaw_joint": 0.0,
  "left_hip_roll_joint": 0.0,
  "left_hip_pitch_joint": -0.1,
  "left_knee_joint": 0.3,
  "left_ankle_pitch_joint": -0.2,
  "left_ankle_roll_joint": 0.0,
  "left_wrist_roll_joint": 0.0,
  "right_hip_yaw_joint": 0.0,
  "right_hip_roll_joint": 0.0,
  "right_hip_pitch_joint": -0.1,
  "right_knee_joint": 0.3,
  "right_ankle_pitch_joint": -0.2,
  "right_ankle_roll_joint": 0.0,
  "right_wrist_roll_joint": 0.0,
  "waist_yaw_joint": 0.0,
  "left_shoulder_pitch_joint": 0.0,
  "left_shoulder_roll_joint": 0.3,
  "left_shoulder_yaw_joint": 0.0,
  "left_elbow_joint": 0.0,
  "right_shoulder_pitch_joint": 0.0,
  "right_shoulder_roll_joint": -0.3,
  "right_shoulder_yaw_joint": 0.0,
  "right_elbow_joint": 0.0,
}
_HOST_GETUP_STYLE_JOINT_NAMES = tuple(_HOST_GETUP_TARGET_JOINT_ANGLES)

_HOST_GETUP_STABLE_SUCCESS_PARAMS = {
  "feet_sensor_name": "feet_ground_contact",
  "body_sensor_name": "support_body_contact",
  "hand_sensor_name": "hand_ground_contact",
  "foot_geom_sensor_name": "foot_geom_ground_contact",
  "min_feet_contact_count": 2.0,
  "max_body_support_count": 0.0,
  "max_hand_contact_count": 0.0,
  "min_foot_flatness": 0.6,
  "min_foot_heading_alignment": 0.6,
  "min_foot_geom_contact_spread": 0.5,
}


def _host_getup_stable_success_params() -> dict:
  """Return fresh stable-success params.

  SceneEntityCfg.resolve() mutates resolved ids in-place during manager setup, so
  reward/metric/event terms must not share the same SceneEntityCfg instances.
  """

  return {
    **_HOST_GETUP_STABLE_SUCCESS_PARAMS,
    "foot_asset_cfg": SceneEntityCfg(
      "robot",
      body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
  }



_HOST_GETUP_TERRAINS = (
  "flat",
  "pyramid_stairs",
  "hf_pyramid_slope",
  "random_rough",
)
GETUP_SINGLE_TERRAIN_VARIANTS = ("ground", "platform", "wall", "slope")
GETUP_TRAIN_MIX_TERRAIN = "mixed"
GETUP_TERRAIN_VARIANTS = (GETUP_TRAIN_MIX_TERRAIN, *GETUP_SINGLE_TERRAIN_VARIANTS)
_HOST_GETUP_HOLDOUT_TERRAINS = (
  "open_stairs",
  "random_stairs",
  "random_spread_boxes",
)


def _set_train_getup_terrain_mix(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = cfg.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.terrain_generator
  assert terrain_generator is not None
  sub_terrains = {
    name: replace(ROUGH_TERRAINS_CFG.sub_terrains[name], proportion=1.0 / len(_HOST_GETUP_TERRAINS))
    for name in _HOST_GETUP_TERRAINS
  }
  terrain.terrain_generator = replace(terrain_generator, sub_terrains=sub_terrains)


def _host_variant_sub_terrains(terrain_variant: str) -> dict:
  parity = HOST_TERRAIN_PARITY[terrain_variant]
  terrain_names = parity["mjlab_terrain_names"]
  proportion = 1.0 / len(terrain_names)
  return {
    name: replace(ROUGH_TERRAINS_CFG.sub_terrains[name], proportion=proportion)
    for name in terrain_names
  }


def _apply_host_terrain_variant(cfg: ManagerBasedRlEnvCfg, terrain_variant: str) -> None:
  if terrain_variant == GETUP_TRAIN_MIX_TERRAIN:
    _set_train_getup_terrain_mix(cfg)
  if terrain_variant not in GETUP_TERRAIN_VARIANTS:
    raise ValueError(
      f"Unsupported Unitree-G1-GetUp terrain {terrain_variant!r}; "
      f"expected one of {GETUP_TERRAIN_VARIANTS}."
    )
  parity = HOST_TERRAIN_PARITY[terrain_variant]
  # Attach explicit metadata for tests, scripts, export/deploy, and mapping docs.
  cfg.getup_terrain = terrain_variant  # type: ignore[attr-defined]
  cfg.host_source_task = HOST_SOURCE_TASKS[terrain_variant]  # type: ignore[attr-defined]
  cfg.host_parity = parity  # type: ignore[attr-defined]
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.max_init_terrain_level = parity["max_init_terrain_level"]
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
      cfg.scene.terrain.terrain_generator = replace(
        generator,
        num_rows=parity["num_rows"],
        num_cols=parity["num_cols"],
        curriculum=parity["curriculum"],
        sub_terrains=_host_variant_sub_terrains(terrain_variant),
      )


def _set_benchmark_holdout_terrain_mix(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = cfg.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.terrain_generator
  assert terrain_generator is not None
  sub_terrains = {
    name: replace(ALL_TERRAINS_CFG.sub_terrains[name], proportion=1.0 / len(_HOST_GETUP_HOLDOUT_TERRAINS))
    for name in _HOST_GETUP_HOLDOUT_TERRAINS
  }
  terrain.terrain_generator = replace(terrain_generator, sub_terrains=sub_terrains, curriculum=False)

def _apply_zero_command_profile(cfg: ManagerBasedRlEnvCfg) -> None:
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = 1.0
  twist_cmd.heading_command = False
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None


def _add_support_depth_camera(cfg: ManagerBasedRlEnvCfg) -> None:
  depth_camera = CameraSensorCfg(
    name="support_depth",
    parent_body="robot/torso_link",
    pos=(0.18, 0.0, 0.24),
    quat=(1.0, 0.0, 0.0, 0.0),
    width=32,
    height=32,
    data_types=("depth",),
    enabled_geom_groups=(0, 3),
    use_textures=False,
    use_shadows=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (depth_camera,)
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "support_depth": ObservationTermCfg(
        func=manipulation_mdp.camera_depth,
        params={"sensor_name": depth_camera.name, "cutoff_distance": 1.5},
      )
    },
    enable_corruption=False,
    concatenate_terms=True,
  )




def _add_support_body_contact_sensor(cfg: ManagerBasedRlEnvCfg) -> None:
  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  hand_geom_names = ("left_hand_collision", "right_hand_collision")
  foot_geom_contact_cfg = ContactSensorCfg(
    name="foot_geom_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r"^(left|right)_foot[1-7]_collision$",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  hand_contact_cfg = ContactSensorCfg(
    name="hand_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r"^(left|right)_hand_collision$",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  support_contact_cfg = ContactSensorCfg(
    name="support_body_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision$",
      exclude=tuple(foot_geom_names + hand_geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    foot_geom_contact_cfg,
    hand_contact_cfg,
    support_contact_cfg,
  )
  cfg.observations["actor"].terms["getup_progress"] = ObservationTermCfg(
    func=mdp.getup_progress_features,
    history_length=6,
    params={
      "sensor_name": support_contact_cfg.name,
      "feet_sensor_name": "feet_ground_contact",
      "hand_sensor_name": hand_contact_cfg.name,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.observations["actor"].terms["bfm_local_body_state"] = ObservationTermCfg(
    func=mdp.bfm_local_body_state,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.observations["critic"].terms["support_contact_pattern"] = ObservationTermCfg(
    func=mdp.support_body_contact_pattern,
    params={"sensor_name": support_contact_cfg.name},
  )
  cfg.observations["critic"].terms["getup_progress"] = ObservationTermCfg(
    func=mdp.getup_progress_features,
    params={
      "sensor_name": support_contact_cfg.name,
      "feet_sensor_name": "feet_ground_contact",
      "hand_sensor_name": hand_contact_cfg.name,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.observations["critic"].terms["bfm_local_body_state"] = ObservationTermCfg(
    func=mdp.bfm_local_body_state,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.metrics["support_body_contact_count"] = MetricsTermCfg(
    func=mdp.support_body_contact_count,
    params={"sensor_name": support_contact_cfg.name},
  )
  cfg.metrics["torso_clearance"] = MetricsTermCfg(
    func=mdp.torso_clearance,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis", "torso_link"))},
  )
  cfg.metrics["getup_upright"] = MetricsTermCfg(
    func=mdp.getup_upright,
    params={"torso_height_threshold": GETUP_SUCCESS_TORSO_HEIGHT, **_host_getup_stable_success_params()},
  )
  cfg.metrics["getup_success_count"] = MetricsTermCfg(
    func=mdp.getup_success_count,
    params={"torso_height_threshold": GETUP_SUCCESS_TORSO_HEIGHT, **_host_getup_stable_success_params()},
  )
  cfg.metrics["getup_latency"] = MetricsTermCfg(
    func=mdp.getup_latency,
    params={"torso_height_threshold": GETUP_SUCCESS_TORSO_HEIGHT, **_host_getup_stable_success_params()},
  )
  cfg.metrics["pelvis_clearance_violation"] = MetricsTermCfg(
    func=mdp.pelvis_clearance_violation,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
  )


def _restore_getup_actor_height_scan(cfg: ManagerBasedRlEnvCfg, *, history_length: int) -> None:
  """Expose terrain geometry to the GetUp actor for one-policy terrain transfer."""

  critic_height_scan = cfg.observations["critic"].terms.get("height_scan")
  if critic_height_scan is None:
    return
  cfg.observations["actor"].terms["height_scan"] = replace(
    critic_height_scan,
    history_length=history_length,
  )



def _move_randomize_terrain_before_root_reset(cfg: ManagerBasedRlEnvCfg) -> None:
  """Ensure play-mode terrain origin changes happen before fallen root reset.

  The base velocity play config appends ``randomize_terrain`` after robot reset
  events. For GetUp on generated terrain that places the fallen robot using the
  previous ``env_origin`` and only then swaps to a new terrain origin, slope and
  wall resets can start meters above or below the sampled patch. BFM-style
  recovery assumes the initial fall state is coherent with the current terrain.
  """

  randomize = cfg.events.get("randomize_terrain")
  if randomize is None or getattr(randomize, "mode", None) != "reset":
    return

  ordered = OrderedDict()
  ordered["randomize_terrain"] = randomize
  for name, term in cfg.events.items():
    if name != "randomize_terrain":
      ordered[name] = term
  cfg.events = ordered



def _add_getup_stall_guard(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.terminations.pop("fell_over", None)
  cfg.terminations.pop("head_contact", None)
  cfg.terminations["stalled_getup"] = TerminationTermCfg(
    func=mdp.stalled_getup_progress,
    params={
      # 600 env steps = 12 seconds at the current 20 ms control step.  Local
      # standardized fall->stand AMP segments include valid recoveries up to
      # about 11.2s; shorter guards terminate both learning and evaluation
      # before a demonstrated get-up can complete.
      "min_steps_before_check": GETUP_STALL_MIN_STEPS,
      "progress_threshold": 0.2,
      "target_height": 0.55,
      "recovery_grace_s": 0.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )


def _apply_getup_nan_safety(cfg: ManagerBasedRlEnvCfg) -> None:
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].nan_policy = "sanitize"
    cfg.observations[group_name].nan_check_per_term = True

  cfg.terminations["unstable_state"] = TerminationTermCfg(
    func=mdp.unstable_getup_state,
    params={
      "max_root_lin_vel": 20.0,
      "max_root_ang_vel": 40.0,
      "max_joint_vel": 120.0,
      "max_joint_acc": 20_000.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )


def _apply_host_effective_action_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["actions"] = replace(
      cfg.observations[group_name].terms["actions"],
      func=mdp.host_effective_actions,
    )


def _apply_host_getup_reward_stack(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace inherited locomotion/recovery shaping with HoST-like get-up terms."""

  regularizer_keep = {
    "body_ang_vel",
    "angular_momentum",
    "joint_acc_l2",
    "joint_pos_limits",
    "action_rate_l2",
    "self_collisions",
    "support_body_contact_penalty_after_lift",
    "pelvis_clearance_penalty",
  }
  for reward_name in list(cfg.rewards):
    if reward_name not in regularizer_keep:
      cfg.rewards.pop(reward_name, None)

  cfg.rewards["host_task_reward"] = RewardTermCfg(
    func=mdp.host_getup_task_reward,
    weight=2.5,
    params={
      **_host_getup_stable_success_params(),
      "orientation_threshold": 0.99,
      "orientation_margin": 0.05,
      "target_base_height_phase1": 0.45,
      "target_base_height_phase3": 0.65,
    },
  )
  # Dense progress signal that survives the fallen-state zero-gradient trap.
  # Without it the multiplicative host_task_reward stays at 0 in the supine
  # start state and PPO has no gradient toward 'lift the torso'.
  cfg.rewards["host_lift_progress"] = RewardTermCfg(
    func=mdp.host_getup_lift_progress_reward,
    weight=1.5,
    params={
      "min_height": 0.12,
      "target_height": 0.55,
      # Require progress toward upright while still keeping a dense gradient
      # once the torso starts rotating in the correct direction.  A -1.0 floor
      # overpaid high-but-sideways postures and never forced the final turn.
      "orientation_floor": 0.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_upright_progress"] = RewardTermCfg(
    func=mdp.host_upright_progress_reward,
    weight=1.5,
    params={
      "min_height": 0.18,
      "target_height": GETUP_SUCCESS_TORSO_HEIGHT,
      # Keep the dense rotation reward focused on real get-up progress.  The
      # failed play-like policies reached ~0.50m but stayed only weakly upright;
      # rewarding alignment below zero would again pay sideways/fallen poses.
      "alignment_floor": 0.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_support_relief"] = RewardTermCfg(
    func=mdp.host_support_relief_reward,
    weight=1.5,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "min_height": 0.18,
      "target_height": GETUP_SUCCESS_TORSO_HEIGHT,
      "max_body_support_count": 8.0,
      "alignment_floor": 0.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_action_smoothness"] = RewardTermCfg(
    func=mdp.host_action_smoothness_penalty,
    weight=-0.01,
    params={"action_rate_weight": 1.0, "smoothness_weight": 1.0},
  )
  cfg.rewards["host_joint_tracking"] = RewardTermCfg(
    func=mdp.host_joint_tracking_penalty,
    weight=-0.00025,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["host_style_pose"] = RewardTermCfg(
    func=mdp.host_style_pose_reward,
    weight=0.8,
    params={
      "joint_names": _HOST_GETUP_STYLE_JOINT_NAMES,
      "target_joint_angles": dict(_HOST_GETUP_TARGET_JOINT_ANGLES),
      "std": 0.6,
      "min_height": GETUP_SUCCESS_TORSO_HEIGHT,
      "min_alignment": 0.75,
      "asset_cfg": SceneEntityCfg("robot"),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_feet_support"] = RewardTermCfg(
    func=mdp.host_feet_support_reward,
    weight=1.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "max_body_support_count": 2.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_hand_support_progress"] = RewardTermCfg(
    func=mdp.host_hand_support_progress_reward,
    weight=1.0,
    params={
      "hand_sensor_name": "hand_ground_contact",
      "min_height": 0.18,
      "release_height": 0.55,
      "final_upright_threshold": 0.9,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_hand_push"] = RewardTermCfg(
    func=mdp.host_hand_push_reward,
    weight=1.2,
    params={
      "hand_sensor_name": "hand_ground_contact",
      "min_height": 0.18,
      "release_height": 0.55,
      "vertical_velocity_scale": 0.5,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_hand_contact_after_stand"] = RewardTermCfg(
    func=mdp.host_hand_contact_after_stand_penalty,
    weight=-1.0,
    params={
      "hand_sensor_name": "hand_ground_contact",
      "activation_height": GETUP_SUCCESS_TORSO_HEIGHT,
      "upright_alignment_threshold": 0.9,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_foot_contact_spread"] = RewardTermCfg(
    func=mdp.host_foot_contact_spread_reward,
    weight=2.0,
    params={
      "foot_geom_sensor_name": "foot_geom_ground_contact",
      "feet_sensor_name": "feet_ground_contact",
      "min_height": 0.35,
      "target_contacts_per_foot": 3.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_foot_flat"] = RewardTermCfg(
    func=mdp.host_foot_flat_reward,
    weight=3.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "min_height": 0.45,
      "min_alignment": 0.5,
      "foot_asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_foot_heading"] = RewardTermCfg(
    func=mdp.host_foot_heading_reward,
    weight=1.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "min_height": 0.45,
      "min_alignment": 0.5,
      "foot_asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_natural_stand_pose"] = RewardTermCfg(
    func=mdp.host_natural_stand_pose_reward,
    weight=2.5,
    params={
      "joint_names": _HOST_GETUP_STYLE_JOINT_NAMES,
      "target_joint_angles": dict(_HOST_GETUP_TARGET_JOINT_ANGLES),
      "std": 0.35,
      "min_height": GETUP_SUCCESS_TORSO_HEIGHT,
      "min_alignment": 0.75,
      "asset_cfg": SceneEntityCfg("robot"),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_foot_orientation_penalty"] = RewardTermCfg(
    func=mdp.host_foot_orientation_penalty,
    weight=-0.8,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "min_height": 0.50,
      "min_alignment": 0.75,
      "foot_asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_ankle_deviation_penalty"] = RewardTermCfg(
    func=mdp.host_ankle_deviation_penalty,
    weight=-1.0,
    params={
      "joint_names": (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
      ),
      "target_joint_angles": {
        "left_ankle_pitch_joint": _HOST_GETUP_TARGET_JOINT_ANGLES["left_ankle_pitch_joint"],
        "left_ankle_roll_joint": _HOST_GETUP_TARGET_JOINT_ANGLES["left_ankle_roll_joint"],
        "right_ankle_pitch_joint": _HOST_GETUP_TARGET_JOINT_ANGLES["right_ankle_pitch_joint"],
        "right_ankle_roll_joint": _HOST_GETUP_TARGET_JOINT_ANGLES["right_ankle_roll_joint"],
      },
      "std": 0.35,
      "min_height": 0.50,
      "min_alignment": 0.75,
      "asset_cfg": SceneEntityCfg("robot"),
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["host_target_standing"] = RewardTermCfg(
    func=mdp.host_target_standing_reward,
    weight=2.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "base_height_target": 0.75,
      "target_base_height_phase3": 0.65,
      "standing_gate_start_height": 0.45,
      "max_body_support_count": 8.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["getup_completion_bonus"] = RewardTermCfg(
    func=mdp.getup_completion_bonus,
    weight=5.0,
    params={
      "tilt_threshold": 0.3,
      "torso_height_threshold": GETUP_SUCCESS_TORSO_HEIGHT,
      **_host_getup_stable_success_params(),
    },
  )


def _make_g1_getup_env_cfg(terrain: str = GETUP_TRAIN_MIX_TERRAIN, play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_23dof_rough_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = GETUP_TRAIN_NUM_ENVS
  _apply_antifall_actor_contract(cfg, history_length=6)
  actor_obs = cfg.observations["actor"]
  actor_history_length = actor_obs.history_length
  actor_obs.history_length = None
  for term in actor_obs.terms.values():
    term.history_length = int(actor_history_length or 1)
  cfg.host_unactuated_timesteps = _HOST_GETUP_UNACTUATED_TIMESTEPS  # type: ignore[attr-defined]
  cfg.host_reward_groups = ("task", "regu", "style", "target")  # type: ignore[attr-defined]
  cfg.actions["joint_pos"] = HostRelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    unactuated_timesteps=_HOST_GETUP_UNACTUATED_TIMESTEPS,
    max_delta=_HOST_GETUP_MAX_ACTION_DELTA,
  )
  _move_randomize_terrain_before_root_reset(cfg)
  _apply_zero_command_profile(cfg)
  _apply_host_terrain_variant(cfg, terrain)
  cfg.curriculum.pop("command_vel", None)
  _add_support_depth_camera(cfg)
  _add_support_body_contact_sensor(cfg)
  _restore_getup_actor_height_scan(cfg, history_length=int(actor_history_length or 1))
  _add_getup_stall_guard(cfg)
  _apply_getup_nan_safety(cfg)
  _apply_host_effective_action_observations(cfg)
  cfg.episode_length_s = GETUP_EPISODE_LENGTH_S
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
  cfg.events.pop("push_robot", None)
  if not play:
    cfg.events["getup_assist_force"] = EventTermCfg(
      func=mdp.apply_host_getup_assist_force,
      mode="step",
      params={
        "initial_force_n": _getup_initial_assist_force(terrain),
        "initial_action_scale": _HOST_GETUP_INITIAL_ACTION_SCALE,
        "success_height_threshold": GETUP_SUCCESS_TORSO_HEIGHT,
        "force_decay_n": 20.0,
        "action_scale_decay": 0.0,
        "min_force_n": 0.0,
        "min_action_scale": _HOST_GETUP_INITIAL_ACTION_SCALE,
        "unactuated_timesteps": _HOST_GETUP_UNACTUATED_TIMESTEPS,
        "orientation_projected_gravity_z_max": -0.8,
        # The assist is a get-up curriculum crutch: it must pull during fallen
        # and side-lying exploration after the startup window.  Keeping HoST's
        # strict upright-orientation gate here makes the force inactive until
        # the robot is already almost standing, so learning never reaches the
        # success/decay path.  Stable support is still required before decay.
        "no_orientation_gate": True,
        "stable_success_required": True,
        "upright_alignment_threshold": 0.85,
        **_host_getup_stable_success_params(),
        # Taper the crutch before the success band.  This keeps the vertical
        # pull useful for early exploration but prevents train-only external
        # force from carrying the policy through the actual upright success.
        "taper_start_height": 0.35,
        "taper_end_height": GETUP_SUCCESS_TORSO_HEIGHT,
        # Keep the final fine-tune close to evaluation dynamics.  The previous
        # assist-heavy schedule made train-like diagnostics succeed on complex
        # terrain while play-like/no-assist rollouts stayed around 0.6 success.
        # Bias episodes toward BFM-style unassisted recovery and keep the action
        # scale identical to play mode.
        "no_assist_probability_initial": 0.05,
        "no_assist_probability": 0.8,
        "no_assist_ramp_start_progress": 0.5,
        "no_assist_ramp_end_progress": 1.0,
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    )
    cfg.metrics["getup_assist_force_n"] = MetricsTermCfg(
      func=mdp.getup_assist_force_n,
      params={"initial_force_n": _getup_initial_assist_force(terrain)},
    )
    cfg.metrics["getup_action_rescale"] = MetricsTermCfg(
      func=mdp.getup_action_rescale,
      params={"initial_action_scale": _HOST_GETUP_INITIAL_ACTION_SCALE},
    )
  cfg.events["reset_robot_joints"].func = mdp.reset_joints_from_presets
  cfg.events["reset_robot_joints"].params = {
    "position_noise_range": (-0.05, 0.05),
    "velocity_range": (-0.5, 0.5),
    "asset_cfg": SceneEntityCfg("robot"),
  }
  _apply_antifall_helpers(
    cfg,
    hard_reset_prob=1.0,
    hard_pose_range=_GETUP_HARD_POSE_RANGE,
    hard_velocity_range=_GETUP_HARD_VELOCITY_RANGE,
  )
  cfg.rewards["body_ang_vel"].func = mdp.bounded_body_angular_velocity_penalty
  cfg.rewards["body_ang_vel"].params.update(
    {
      "max_penalty": 400.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    }
  )
  cfg.rewards["angular_momentum"].func = mdp.bounded_angular_momentum_penalty
  cfg.rewards["angular_momentum"].params["max_penalty"] = 1000.0
  cfg.rewards["joint_acc_l2"].func = mdp.bounded_joint_acc_l2
  cfg.rewards["joint_acc_l2"].params = {
    "max_penalty": 1_000_000.0,
    "asset_cfg": SceneEntityCfg("robot"),
  }
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.bounded_action_rate_after_lift,
    weight=-0.05,
    params={
      "activation_height": 0.25,
      "max_penalty": 250.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=mdp.joint_pos_limits_after_support,
    weight=-10.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "max_penalty": 10.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost_after_support,
    weight=-1.0,
    params={
      "sensor_name": "self_collision",
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["support_body_contact_penalty_after_lift"] = RewardTermCfg(
    func=mdp.support_body_contact_penalty_after_lift,
    weight=-0.75,
    params={
      "sensor_name": "support_body_contact",
      "hand_sensor_name": "hand_ground_contact",
      "activation_height": 0.2,
      "hand_release_height": 0.55,
      "normalize_count": 2.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["pelvis_clearance_penalty"] = RewardTermCfg(
    func=mdp.pelvis_clearance_penalty,
    weight=-1.0,
  )
  cfg.rewards.pop("is_terminated", None)
  cfg.events["reset_base"].func = mdp.reset_root_state_from_presets
  cfg.events["reset_base"].params = {
    "presets": (
      {
        "name": "supine",
        "pose_range": {
          "x": (-0.15, 0.15),
          "y": (-0.15, 0.15),
          # reset_root_state_uniform() is relative to the standing default root pose.
          # Supine uses a higher fallen-but-not-standing z-offset than side
          # resets: the lower side-lying offset penetrates the supine torso/limb
          # stack and creates an upward contact impulse before policy action.
          "z": (-0.35, -0.25),
          "roll": (3.14159 - 0.3, 3.14159 + 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-3.14159, 3.14159),
        },
      },
      {
        "name": "left_side",
        "pose_range": {
          "x": (-0.15, 0.15),
          "y": (-0.15, 0.15),
          "z": (-0.7, -0.6),
          "roll": (1.5708 - 0.25, 1.5708 + 0.25),
          "pitch": (-0.35, 0.35),
          "yaw": (-3.14159, 3.14159),
        },
      },
      {
        "name": "right_side",
        "pose_range": {
          "x": (-0.15, 0.15),
          "y": (-0.15, 0.15),
          "z": (-0.7, -0.6),
          "roll": (-1.5708 - 0.25, -1.5708 + 0.25),
          "pitch": (-0.35, 0.35),
          "yaw": (-3.14159, 3.14159),
        },
      },
      {
        "name": "seated_fall",
        "pose_range": {
          "x": (-0.15, 0.15),
          "y": (-0.15, 0.15),
          "z": (-0.5, -0.4),
          "roll": (-0.2, 0.2),
          "pitch": (1.5708 - 0.35, 1.5708 + 0.35),
          "yaw": (-3.14159, 3.14159),
        },
      },
    ),
    "preset_weight_stages": _GETUP_PLAY_PRESET_WEIGHT_STAGES if play else _GETUP_TRAIN_PRESET_WEIGHT_STAGES,
    "velocity_range": _GETUP_HARD_VELOCITY_RANGE,
  }
  _apply_host_getup_reward_stack(cfg)
  return cfg



def _apply_getup_amp_observation_group(
  cfg: ManagerBasedRlEnvCfg,
  *,
  demo_data_dir: str,
) -> None:
  """Attach discriminator-only AMP observations to an opt-in GetUp config."""
  cfg.getup_amp_enabled = True  # type: ignore[attr-defined]
  cfg.getup_amp_demo_data_dir = demo_data_dir  # type: ignore[attr-defined]
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "features": ObservationTermCfg(
        func=mdp.amp_getup_features,
        params={"asset_cfg": SceneEntityCfg("robot")},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
    nan_policy="sanitize",
    nan_check_per_term=True,
  )


def unitree_g1_getup_env_cfg(terrain: str = GETUP_TRAIN_MIX_TERRAIN, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 HoST get-up configuration for one terrain variant."""
  return _make_g1_getup_env_cfg(terrain=terrain, play=play)


def unitree_g1_getup_amp_env_cfg(
  demo_data_dir: str = "data/motions/g1_getup_amp",
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the opt-in ground-only G1 GetUp AMP fallback env config.

  The default HoST-parity Unitree-G1-GetUp task remains no-demo.  AMP is
  intentionally constrained to flat ground in this first pass.
  """
  cfg = _make_g1_getup_env_cfg(terrain="ground", play=play)
  cfg.getup_terrain = "ground"  # type: ignore[attr-defined]
  _apply_getup_amp_observation_group(cfg, demo_data_dir=demo_data_dir)
  return cfg


def unitree_g1_getup_benchmark_env_cfg(terrain: str = GETUP_TRAIN_MIX_TERRAIN, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a deterministic HoST get-up benchmark scaffold."""
  cfg = _make_g1_getup_env_cfg(terrain=terrain, play=play)
  _set_benchmark_holdout_terrain_mix(cfg)
  cfg.curriculum = {}
  cfg.observations["actor"].enable_corruption = False
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (0.0, 0.0),
    1: (0.0, 0.0),
    2: (0.0, 0.0),
  }
  return cfg
