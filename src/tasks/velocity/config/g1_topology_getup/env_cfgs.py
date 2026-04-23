"""Terrain get-up environment scaffolds for Unitree G1.

This family is intentionally isolated from the existing flat anti-fall tasks so the
new mandatory-depth get-up work does not change current training task behavior.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from src.tasks.velocity.config.g1_antifall.env_cfgs import (
  _apply_antifall_actor_contract,
  _apply_antifall_helpers,
)

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
_DEFAULT_GETUP_DEMO_NPZ = str(Path("src/assets/motions/g1/getup_synthetic_demo.npz"))



_SEEN_TOPOLOGY_TERRAINS = (
  "flat",
  "pyramid_stairs",
  "hf_pyramid_slope",
  "random_rough",
)
_HOLDOUT_TOPOLOGY_TERRAINS = (
  "open_stairs",
  "random_stairs",
  "random_spread_boxes",
)


def _set_train_topology_terrain_mix(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = cfg.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.terrain_generator
  assert terrain_generator is not None
  sub_terrains = {
    name: replace(ROUGH_TERRAINS_CFG.sub_terrains[name], proportion=1.0 / len(_SEEN_TOPOLOGY_TERRAINS))
    for name in _SEEN_TOPOLOGY_TERRAINS
  }
  terrain.terrain_generator = replace(terrain_generator, sub_terrains=sub_terrains)


def _set_benchmark_holdout_terrain_mix(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain = cfg.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.terrain_generator
  assert terrain_generator is not None
  sub_terrains = {
    name: replace(ALL_TERRAINS_CFG.sub_terrains[name], proportion=1.0 / len(_HOLDOUT_TOPOLOGY_TERRAINS))
    for name in _HOLDOUT_TOPOLOGY_TERRAINS
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
  support_contact_cfg = ContactSensorCfg(
    name="support_body_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision$",
      exclude=tuple(foot_geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (support_contact_cfg,)
  cfg.observations["actor"].terms["getup_progress"] = ObservationTermCfg(
    func=mdp.getup_progress_features,
    params={
      "sensor_name": support_contact_cfg.name,
      "feet_sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
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
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
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
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.metrics["getup_success_count"] = MetricsTermCfg(
    func=mdp.getup_success_count,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.metrics["getup_latency"] = MetricsTermCfg(
    func=mdp.getup_latency,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.metrics["pelvis_clearance_violation"] = MetricsTermCfg(
    func=mdp.pelvis_clearance_violation,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
  )



def _add_head_contact_guard(cfg: ManagerBasedRlEnvCfg) -> None:
  head_contact_cfg = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=("head_collision",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (head_contact_cfg,)
  cfg.terminations.pop("fell_over", None)
  cfg.terminations["head_contact"] = TerminationTermCfg(
    func=mdp.tolerant_illegal_contact,
    params={
      "sensor_name": head_contact_cfg.name,
      "force_threshold": 5.0,
      "bad_contact_time_threshold_s": 0.5,
      "grace_period_s": 1.2,
    },
  )
  cfg.terminations["stalled_getup"] = TerminationTermCfg(
    func=mdp.stalled_getup_progress,
    params={
      "min_steps_before_check": 50,
      "progress_threshold": 0.2,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )

def _make_g1_topology_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_rough_env_cfg(play=play)
  _apply_antifall_actor_contract(cfg)
  _apply_zero_command_profile(cfg)
  _set_train_topology_terrain_mix(cfg)
  cfg.curriculum.pop("command_vel", None)
  _add_support_depth_camera(cfg)
  _add_support_body_contact_sensor(cfg)
  _add_head_contact_guard(cfg)
  cfg.episode_length_s = 20.0
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 128)
  cfg.events.pop("push_robot", None)
  if not play:
    cfg.events["getup_assist_force"] = EventTermCfg(
      func=mdp.apply_getup_assist_force,
      mode="step",
      params={
        "force_n": 75.0,
        "activation_height": 0.35,
        "alignment_threshold": 0.0,
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
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
  cfg.rewards["getup_posture_reward"] = RewardTermCfg(
    func=mdp.getup_posture_reward,
    weight=1.5,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity_after_lift,
    weight=1.0,
    params={
      "std": 1.0,
      "command_name": "twist",
      "activation_height": 0.45,
      "alignment_threshold": 0.3,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_angular_velocity_after_lift,
    weight=1.0,
    params={
      "std": 1.0,
      "command_name": "twist",
      "activation_height": 0.45,
      "alignment_threshold": 0.3,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["getup_torso_lift_reward"] = RewardTermCfg(
    func=mdp.getup_torso_lift_reward,
    weight=3.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.rewards["getup_facing_up_reward"] = RewardTermCfg(
    func=mdp.getup_facing_up_reward,
    weight=3.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.rewards["getup_orientation_phase_bonus"] = RewardTermCfg(
    func=mdp.getup_orientation_phase_bonus,
    weight=4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "thresholds": (0.1, 0.4, 0.7),
      "bonuses": (1.0, 2.0, 3.0),
    },
  )
  cfg.rewards["getup_height_progress_reward"] = RewardTermCfg(
    func=mdp.getup_height_progress_reward,
    weight=0.75,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
  )
  cfg.rewards["getup_phase_bonus"] = RewardTermCfg(
    func=mdp.getup_phase_bonus,
    weight=10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "thresholds": (0.22, 0.4, 0.55),
      "bonuses": (1.0, 2.0, 3.0),
    },
  )
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still_after_getup,
    weight=-1.0,
    params={
      "command_name": "twist",
      "activation_height": 0.45,
      "facing_up_threshold": 0.3,
      "torso_asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "joint_asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_after_lift,
    weight=-0.05,
    params={
      "activation_height": 0.25,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=mdp.joint_pos_limits_after_support,
    weight=-10.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
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
  cfg.rewards["support_contact_diversity_reward"] = RewardTermCfg(
    func=mdp.support_contact_diversity_reward,
    weight=0.3,
    params={
      "sensor_name": "support_body_contact",
      "active_below_height": 0.2,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["support_body_contact_penalty_after_lift"] = RewardTermCfg(
    func=mdp.support_body_contact_penalty_after_lift,
    weight=-0.75,
    params={
      "sensor_name": "support_body_contact",
      "activation_height": 0.2,
      "normalize_count": 2.0,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["getup_feet_support_reward"] = RewardTermCfg(
    func=mdp.getup_feet_support_reward,
    weight=1.5,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["getup_standing_joint_pose_reward"] = RewardTermCfg(
    func=mdp.getup_standing_joint_pose_reward,
    weight=2.0,
    params={
      "feet_sensor_name": "feet_ground_contact",
      "body_sensor_name": "support_body_contact",
      "joint_names": (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "waist_pitch_joint",
      ),
    },
  )
  cfg.rewards["getup_demo_pose_reward"] = RewardTermCfg(
    func=mdp.getup_demo_pose_reward,
    weight=1.0,
    params={
      "demo_npz_path": _DEFAULT_GETUP_DEMO_NPZ,
      "joint_names": (
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "waist_pitch_joint",
      ),
      "dt_per_demo_frame": 0.02,
    },
  )
  cfg.rewards["reduced_support_bonus"] = RewardTermCfg(
    func=mdp.reduced_support_bonus,
    weight=5.0,
    params={
      "sensor_name": "support_body_contact",
      "max_support_count": 0.5,
      "activation_height": 0.4,
      "alignment_threshold": 0.3,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.rewards["pelvis_clearance_penalty"] = RewardTermCfg(
    func=mdp.pelvis_clearance_penalty,
    weight=-1.0,
  )
  cfg.rewards["getup_completion_bonus"] = RewardTermCfg(
    func=mdp.getup_completion_bonus,
    weight=2.0,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
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
          # Use negative z-offsets so fallen presets start on terrain instead of at
          # standing height with a long passive drop before recovery begins.
          "z": (-0.7, -0.6),
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
    "preset_weight_stages": (
      {"step": 0, "weights": (0.0, 0.25, 0.25, 0.5)},
      {"step": 48, "weights": (0.15, 0.25, 0.25, 0.35)},
      {"step": 120, "weights": (0.25, 0.25, 0.25, 0.25)},
    ),
    "velocity_range": _GETUP_HARD_VELOCITY_RANGE,
  }
  return cfg


def unitree_g1_topology_getup_stage0_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Stage 0 terrain get-up scaffold with mandatory onboard depth."""
  return _make_g1_topology_getup_env_cfg(play=play)


def unitree_g1_topology_getup_benchmark_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a deterministic terrain get-up benchmark scaffold."""
  cfg = _make_g1_topology_getup_env_cfg(play=play)
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
