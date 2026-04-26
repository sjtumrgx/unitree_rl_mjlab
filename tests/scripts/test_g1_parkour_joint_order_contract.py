from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_YAML_PATH = ROOT / 'deploy' / 'robots' / 'g1_parkour' / 'config' / 'policy' / 'parkour' / 'v0' / 'params' / 'deploy.yaml'
PROMOTE_SCRIPT_PATH = ROOT / 'scripts' / 'promote_g1_parkour_artifact.py'
TRAINING_URDF_PATH = Path('/home/eilab/instinctlab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf')
PARKOUR_SCENE_PATHS = (
  ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1_parkour.xml",
  ROOT / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1_parkour_flat.xml",
)


def _training_joint_order() -> list[str]:
  root = ET.parse(TRAINING_URDF_PATH).getroot()
  return [
    joint.attrib['name']
    for joint in root.findall('joint')
    if joint.attrib.get('type') in {'revolute', 'continuous'}
  ]


def test_parkour_deploy_joint_ids_follow_training_urdf_joint_order() -> None:
  payload = yaml.safe_load(DEPLOY_YAML_PATH.read_text())
  expected = list(range(len(_training_joint_order())))
  assert payload['joint_ids_map'] == expected


def test_parkour_promotion_script_uses_training_joint_order_as_deploy_order() -> None:
  text = PROMOTE_SCRIPT_PATH.read_text()
  assert 'TRAINING_JOINT_NAMES' in text
  assert 'SIM_JOINT_NAMES' not in text


def test_mujoco_play_uses_isaac_onnx_order_distinct_from_deploy_motor_order() -> None:
  from src.parkour.contract import ONNX_POLICY_JOINT_NAMES, TRAINING_JOINT_NAMES

  assert len(ONNX_POLICY_JOINT_NAMES) == len(TRAINING_JOINT_NAMES) == 29
  assert set(ONNX_POLICY_JOINT_NAMES) == set(TRAINING_JOINT_NAMES)
  assert ONNX_POLICY_JOINT_NAMES != TRAINING_JOINT_NAMES
  assert ONNX_POLICY_JOINT_NAMES[:3] == (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
  )


def test_parkour_mujoco_ankle_motor_limits_are_bilateral_and_match_training_effort() -> None:
  """Prevent left/right ankle torque clipping asymmetry in native MuJoCo DDS runs."""
  training = ET.parse(TRAINING_URDF_PATH).getroot()
  expected_effort = {
    joint.attrib["name"]: float(joint.find("limit").attrib["effort"])
    for joint in training.findall("joint")
    if joint.attrib.get("name") in {
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
    }
  }
  assert expected_effort == {
    "left_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 50.0,
    "right_ankle_pitch_joint": 50.0,
    "right_ankle_roll_joint": 50.0,
  }

  for scene_path in PARKOUR_SCENE_PATHS:
    scene = ET.parse(scene_path).getroot()
    motors = {
      motor.attrib["joint"]: tuple(float(part) for part in motor.attrib["ctrlrange"].split())
      for motor in scene.findall("./actuator/motor")
      if motor.attrib["joint"] in expected_effort
    }
    assert set(motors) == set(expected_effort)
    for joint_name, effort in expected_effort.items():
      assert motors[joint_name] == (-effort, effort)
