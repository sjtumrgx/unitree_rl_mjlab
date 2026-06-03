from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import yaml

DEFAULT_DESTINATION_ROOT = Path('deploy/robots/g1_parkour/config/policy/parkour/v0')
DEFAULT_RUN_DIR = Path('/home/eilab/instinctlab/logs/instinct_rl/g1_parkour/20260327_163647')
MANIFEST_NAME = 'parkour_artifacts.json'
POINTCLOUD_TOPIC = 'rt/parkour_depth/points'
DEPTH_TIMEOUT_MS = 500
TRAINING_JOINT_NAMES = [
  'left_hip_pitch_joint',
  'left_hip_roll_joint',
  'left_hip_yaw_joint',
  'left_knee_joint',
  'left_ankle_pitch_joint',
  'left_ankle_roll_joint',
  'right_hip_pitch_joint',
  'right_hip_roll_joint',
  'right_hip_yaw_joint',
  'right_knee_joint',
  'right_ankle_pitch_joint',
  'right_ankle_roll_joint',
  'waist_yaw_joint',
  'waist_roll_joint',
  'waist_pitch_joint',
  'left_shoulder_pitch_joint',
  'left_shoulder_roll_joint',
  'left_shoulder_yaw_joint',
  'left_elbow_joint',
  'left_wrist_roll_joint',
  'left_wrist_pitch_joint',
  'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint',
  'right_shoulder_roll_joint',
  'right_shoulder_yaw_joint',
  'right_elbow_joint',
  'right_wrist_roll_joint',
  'right_wrist_pitch_joint',
  'right_wrist_yaw_joint',
]
PROPRIO_TERM_SPECS = (
  ('base_ang_vel', 'base_ang_vel', 3),
  ('projected_gravity', 'projected_gravity', 3),
  ('velocity_commands', 'velocity_commands', 3),
  ('joint_pos_rel', 'joint_pos', len(TRAINING_JOINT_NAMES)),
  ('joint_vel_rel', 'joint_vel', len(TRAINING_JOINT_NAMES)),
  ('last_action', 'actions', len(TRAINING_JOINT_NAMES)),
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      'Promote a trained InstinctLab G1 parkour artifact bundle into the dedicated '\
      'deploy/robots/g1_parkour runtime staging tree.'
    )
  )
  parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
  parser.add_argument('--destination-root', type=Path, default=DEFAULT_DESTINATION_ROOT)
  return parser


def _copy(src: Path, dst: Path) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)


def _load_yaml(path: Path) -> dict[str, Any]:
  return yaml.unsafe_load(path.read_text())


def _as_list(value: Any) -> list[Any]:
  if isinstance(value, list):
    return value
  if isinstance(value, tuple):
    return list(value)
  return [value]


def _match_value(joint_name: str, raw_value: Any) -> float:
  if isinstance(raw_value, dict):
    for pattern, candidate in raw_value.items():
      if re.fullmatch(pattern, joint_name):
        return float(candidate)
    raise KeyError(f'No value match for joint {joint_name!r}')
  return float(raw_value)


def _joint_vector_from_regex_map(raw_value: Any, *, default: float = 0.0) -> list[float]:
  values: list[float] = []
  for joint_name in TRAINING_JOINT_NAMES:
    try:
      values.append(_match_value(joint_name, raw_value))
    except KeyError:
      values.append(default)
  return values


def _default_joint_pos(env_cfg: dict[str, Any]) -> list[float]:
  joint_pos_cfg = env_cfg['scene']['robot']['init_state']['joint_pos']
  return _joint_vector_from_regex_map(joint_pos_cfg, default=0.0)


def _joint_ids_map() -> list[int]:
  return list(range(len(TRAINING_JOINT_NAMES)))


def _joint_prop_from_actuators(env_cfg: dict[str, Any], field_name: str) -> list[float]:
  actuators = env_cfg['scene']['robot']['actuators']
  values = [None] * len(TRAINING_JOINT_NAMES)
  for actuator_cfg in actuators.values():
    joint_exprs = actuator_cfg.get('joint_names_expr', [])
    field_value = actuator_cfg.get(field_name)
    if field_value is None:
      continue
    for index, joint_name in enumerate(TRAINING_JOINT_NAMES):
      if values[index] is not None:
        continue
      if any(re.fullmatch(expr, joint_name) for expr in joint_exprs):
        values[index] = _match_value(joint_name, field_value)
  if any(value is None for value in values):
    missing = [TRAINING_JOINT_NAMES[i] for i, value in enumerate(values) if value is None]
    raise KeyError(f'Missing {field_name} values for joints: {missing}')
  return [float(value) for value in values]


def _term_scale(term_cfg: dict[str, Any], dim: int) -> list[float]:
  scale = term_cfg.get('scale')
  if scale is None:
    return [1.0] * dim
  if isinstance(scale, (int, float)):
    return [float(scale)] * dim
  scale_list = [float(item) for item in _as_list(scale)]
  if len(scale_list) != dim:
    raise ValueError(f'Expected scale length {dim}, got {len(scale_list)}')
  return scale_list


def _build_proprio_observations(env_cfg: dict[str, Any]) -> dict[str, Any]:
  policy_obs = env_cfg['observations']['policy']
  observations: dict[str, Any] = {}
  for deploy_name, source_name, dim in PROPRIO_TERM_SPECS:
    term_cfg = policy_obs[source_name]
    params = dict(term_cfg.get('params') or {})
    if deploy_name == 'velocity_commands':
      params = {'command_name': 'base_velocity'}
    observations[deploy_name] = {
      'params': params,
      'clip': None,
      'scale': _term_scale(term_cfg, dim),
      'history_length': int(term_cfg['history_length']),
    }
  return observations


def _extract_depth_contract(env_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> dict[str, Any]:
  policy_depth = env_cfg['observations']['policy']['depth_image']
  scene_camera = env_cfg['scene']['camera']
  crop_region = _as_list(scene_camera['noise_pipeline']['crop_and_resize']['crop_region'])
  depth_range = _as_list(scene_camera['noise_pipeline']['depth_normalization']['depth_range'])
  output_range = _as_list(scene_camera['noise_pipeline']['depth_normalization']['output_range'])
  raw_width = int(scene_camera['pattern_cfg']['width'])
  raw_height = int(scene_camera['pattern_cfg']['height'])
  output_width = raw_width - int(crop_region[2]) - int(crop_region[3])
  output_height = raw_height - int(crop_region[0]) - int(crop_region[1])
  history_source_length = int(scene_camera['data_histories']['distance_to_image_plane_noised'])
  history_skip_frames = int(policy_depth['params']['history_skip_frames'])
  num_output_frames = int(policy_depth['params']['num_output_frames'])
  expected_size = num_output_frames * output_height * output_width
  latent_size = int(agent_cfg['policy']['encoder_configs']['depth_encoder']['output_size'])
  return {
    'sensor_name': 'depth_image',
    'camera_name': 'parkour_depth_camera',
    'topic_name': POINTCLOUD_TOPIC,
    'pointcloud_mode': 'z_depth',
    'organized_pointcloud': True,
    'retain_last_valid_frame': True,
    'timeout_ms': DEPTH_TIMEOUT_MS,
    'pointcloud_field_names': {'x': 'x', 'y': 'y', 'z': 'z'},
    'raw_resolution': [raw_width, raw_height],
    'crop_region': [int(v) for v in crop_region],
    'output_resolution': [output_width, output_height],
    'depth_range': [float(v) for v in depth_range],
    'output_range': [float(v) for v in output_range],
    'history_source_length': history_source_length,
    'history_skip_frames': history_skip_frames,
    'num_output_frames': num_output_frames,
    'expected_size': expected_size,
    'latent_size': latent_size,
    'camera_pose': {
      'pos': [float(v) for v in _as_list(scene_camera['offset']['pos'])],
      'rot': [float(v) for v in _as_list(scene_camera['offset']['rot'])],
      'convention': scene_camera['offset']['convention'],
      'parent_prim_path': scene_camera['prim_path'],
    },
  }


def _extract_keyboard_command_limits(env_cfg: dict[str, Any]) -> dict[str, float]:
  base_velocity_cfg = env_cfg['commands']['base_velocity']
  velocity_ranges = base_velocity_cfg.get('velocity_ranges') or {}
  if velocity_ranges:
    lin_vel_x_min = min(float(cfg['lin_vel_x'][0]) for cfg in velocity_ranges.values())
    lin_vel_x_max = max(float(cfg['lin_vel_x'][1]) for cfg in velocity_ranges.values())
    ang_vel_z_min = min(float(cfg['ang_vel_z'][0]) for cfg in velocity_ranges.values())
    ang_vel_z_max = max(float(cfg['ang_vel_z'][1]) for cfg in velocity_ranges.values())
  else:
    ranges = base_velocity_cfg['ranges']
    lin_vel_x_min, lin_vel_x_max = (float(v) for v in _as_list(ranges['lin_vel_x']))
    ang_vel_z_min, ang_vel_z_max = (float(v) for v in _as_list(ranges['ang_vel_z']))
  if base_velocity_cfg.get('only_positive_lin_vel_x'):
    lin_vel_x_min = max(0.0, lin_vel_x_min)
  return {
    'lin_vel_x_min': lin_vel_x_min,
    'lin_vel_x_max': lin_vel_x_max,
    'ang_vel_z_min': ang_vel_z_min,
    'ang_vel_z_max': ang_vel_z_max,
    'lin_vel_step': 0.1,
    'hold_timeout_s': 0.45,
  }


def build_g1_parkour_deploy_cfg(env_cfg: dict[str, Any], agent_cfg: dict[str, Any]) -> dict[str, Any]:
  default_joint_pos = _default_joint_pos(env_cfg)
  stiffness = _joint_prop_from_actuators(env_cfg, 'stiffness')
  damping = _joint_prop_from_actuators(env_cfg, 'damping')
  action_cfg = env_cfg['actions']['joint_pos']
  depth_contract = _extract_depth_contract(env_cfg, agent_cfg)
  proprio_observations = _build_proprio_observations(env_cfg)
  deploy_cfg: dict[str, Any] = {
    'joint_ids_map': _joint_ids_map(),
    'step_dt': float(env_cfg['sim']['dt']) * float(env_cfg['decimation']),
    'stiffness': stiffness,
    'damping': damping,
    'default_joint_pos': default_joint_pos,
    'commands': {
      'base_velocity': {
        'ranges': {
          'lin_vel_x': [float(v) for v in _as_list(env_cfg['commands']['base_velocity']['ranges']['lin_vel_x'])],
          'lin_vel_y': [float(v) for v in _as_list(env_cfg['commands']['base_velocity']['ranges']['lin_vel_y'])],
          'ang_vel_z': [float(v) for v in _as_list(env_cfg['commands']['base_velocity']['ranges']['ang_vel_z'])],
          'heading': None,
        },
        'keyboard': _extract_keyboard_command_limits(env_cfg),
      }
    },
    'actions': {
      'JointPositionAction': {
        'clip': None,
        'joint_names': list(action_cfg['joint_names']),
        'scale': _joint_vector_from_regex_map(action_cfg['scale'], default=1.0),
        'offset': default_joint_pos,
        'joint_ids': None,
      }
    },
    'observations': {
      'proprio': proprio_observations,
      'depth_image': {
        'depth_image': {
          'params': {'sensor_name': 'depth_image', 'expected_size': depth_contract['expected_size']},
          'clip': None,
          'scale': [1.0] * depth_contract['expected_size'],
          'history_length': 1,
        }
      },
    },
    'parkour_depth_interface': depth_contract,
  }
  return deploy_cfg


def promote_g1_parkour_artifact(*, run_dir: Path, destination_root: Path) -> Path:
  source_root = run_dir.expanduser()
  if not source_root.exists():
    raise FileNotFoundError(f'Run directory not found: {source_root}')
  if not source_root.is_dir():
    raise FileNotFoundError(f'Run directory path is not a directory: {source_root}')

  actor_onnx = source_root / 'exported' / 'actor.onnx'
  depth_encoder_onnx = source_root / 'exported' / '0-depth_encoder.onnx'
  env_yaml = source_root / 'params' / 'env.yaml'
  agent_yaml = source_root / 'params' / 'agent.yaml'
  for path in (actor_onnx, depth_encoder_onnx, env_yaml, agent_yaml):
    if not path.exists():
      raise FileNotFoundError(f'Expected artifact file not found: {path}')

  env_cfg = _load_yaml(env_yaml)
  agent_cfg = _load_yaml(agent_yaml)
  deploy_cfg = build_g1_parkour_deploy_cfg(env_cfg, agent_cfg)

  destination = destination_root.expanduser()
  _copy(actor_onnx, destination / 'exported' / 'actor.onnx')
  _copy(depth_encoder_onnx, destination / 'exported' / '0-depth_encoder.onnx')
  _copy(env_yaml, destination / 'params' / 'env.yaml')
  _copy(agent_yaml, destination / 'params' / 'agent.yaml')
  deploy_yaml = destination / 'params' / 'deploy.yaml'
  deploy_yaml.parent.mkdir(parents=True, exist_ok=True)
  deploy_yaml.write_text(yaml.safe_dump(deploy_cfg, sort_keys=False))

  manifest = {
    'schema_version': 'g1_parkour_artifacts_v1',
    'actor_onnx': 'exported/actor.onnx',
    'depth_encoder_onnx': 'exported/0-depth_encoder.onnx',
    'deploy_yaml': 'params/deploy.yaml',
    'env_yaml': 'params/env.yaml',
    'agent_yaml': 'params/agent.yaml',
    'promoted_from_run_dir': str(source_root),
  }
  (destination / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
  return destination


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  promote_g1_parkour_artifact(run_dir=args.run_dir, destination_root=args.destination_root)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
