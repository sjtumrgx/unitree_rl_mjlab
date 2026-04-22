import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.tasks.velocity import mdp

_EXPECTED_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "joint_pos",
  "joint_vel",
  "actions",
)


def test_stage0_keeps_existing_antifall_actor_contract_but_adds_camera_group() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  assert tuple(cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERMS
  assert "camera" in cfg.observations
  assert tuple(cfg.observations["camera"].terms) == ("support_depth",)
  assert "support_contact_pattern" in cfg.observations["critic"].terms
  assert "support_body_contact_count" in cfg.metrics
  assert "torso_clearance" in cfg.metrics
  assert "getup_posture_reward" in cfg.rewards
  assert "support_contact_diversity_reward" in cfg.rewards
  assert "pelvis_clearance_penalty" in cfg.rewards
  assert "getup_completion_bonus" in cfg.rewards
  assert "getup_upright" in cfg.metrics
  assert "getup_success_count" in cfg.metrics
  assert "getup_latency" in cfg.metrics
  assert "pelvis_clearance_violation" in cfg.metrics
  assert any(
    isinstance(sensor, CameraSensorCfg) and sensor.name == "support_depth"
    for sensor in (cfg.scene.sensors or ())
  )
  assert cfg.scene.terrain is not None
  assert "is_terminated" not in cfg.rewards
  assert "fell_over" not in cfg.terminations
  assert "head_contact" in cfg.terminations
  assert cfg.scene.terrain.terrain_type == "generator"
  assert cfg.events["reset_base"].func is mdp.reset_root_state_from_presets
  presets = cfg.events["reset_base"].params["presets"]
  assert tuple(preset["name"] for preset in presets) == ("supine", "left_side", "right_side", "seated_fall")


def test_runner_uses_camera_obs_groups_for_both_actor_and_critic() -> None:
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0")
  assert rl_cfg.obs_groups == {
    "actor": ("actor", "camera"),
    "critic": ("critic", "camera"),
  }


def test_benchmark_disables_randomization_without_mutating_antifall_ids() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Benchmark")
  assert cfg.curriculum == {}
  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.events["foot_friction"].params["ranges"] == (1.0, 1.0)
  antifall_cfg = load_env_cfg("Unitree-G1-AntiFall-Stage0")
  assert tuple(antifall_cfg.observations["actor"].terms) == _EXPECTED_ACTOR_TERMS


def test_runner_uses_topology_bottleneck_model_class() -> None:
  rl_cfg = load_rl_cfg("Unitree-G1-TopologyGetUp-Stage0")
  assert rl_cfg.actor.class_name == "src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel"
  assert rl_cfg.critic.class_name == "src.tasks.velocity.rl.topology_bottleneck_model:TopologyBottleneckCNNModel"
  assert rl_cfg.actor.cnn_cfg["bottleneck_dim"] == 64


def test_stage0_uses_seen_training_terrain_mix_and_disables_command_curriculum() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Stage0")
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  assert tuple(terrain_generator.sub_terrains) == (
    "flat",
    "pyramid_stairs",
    "hf_pyramid_slope",
    "random_rough",
  )
  assert "command_vel" not in cfg.curriculum
  assert "terrain_levels" in cfg.curriculum


def test_benchmark_switches_to_holdout_terrain_mix() -> None:
  cfg = load_env_cfg("Unitree-G1-TopologyGetUp-Benchmark")
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  assert tuple(terrain_generator.sub_terrains) == (
    "open_stairs",
    "random_stairs",
    "random_spread_boxes",
  )
