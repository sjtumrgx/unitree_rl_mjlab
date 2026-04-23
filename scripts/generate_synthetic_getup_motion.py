from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

JOINT_ORDER = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

DEFAULT_STAND = {
  "left_hip_pitch_joint": -0.1,
  "left_hip_roll_joint": 0.0,
  "left_hip_yaw_joint": 0.0,
  "left_knee_joint": 0.3,
  "left_ankle_pitch_joint": -0.2,
  "left_ankle_roll_joint": 0.0,
  "right_hip_pitch_joint": -0.1,
  "right_hip_roll_joint": 0.0,
  "right_hip_yaw_joint": 0.0,
  "right_knee_joint": 0.3,
  "right_ankle_pitch_joint": -0.2,
  "right_ankle_roll_joint": 0.0,
  "waist_yaw_joint": 0.0,
  "waist_roll_joint": 0.0,
  "waist_pitch_joint": 0.0,
  "left_shoulder_pitch_joint": 0.35,
  "left_shoulder_roll_joint": 0.18,
  "left_shoulder_yaw_joint": 0.0,
  "left_elbow_joint": 0.87,
  "left_wrist_roll_joint": 0.0,
  "left_wrist_pitch_joint": 0.0,
  "left_wrist_yaw_joint": 0.0,
  "right_shoulder_pitch_joint": 0.35,
  "right_shoulder_roll_joint": -0.18,
  "right_shoulder_yaw_joint": 0.0,
  "right_elbow_joint": 0.87,
  "right_wrist_roll_joint": 0.0,
  "right_wrist_pitch_joint": 0.0,
  "right_wrist_yaw_joint": 0.0,
}

SUPINE_TUCK = {
  "left_hip_pitch_joint": -1.05,
  "left_knee_joint": 1.85,
  "left_ankle_pitch_joint": -0.85,
  "right_hip_pitch_joint": -1.05,
  "right_knee_joint": 1.85,
  "right_ankle_pitch_joint": -0.85,
  "left_shoulder_pitch_joint": 0.75,
  "left_shoulder_roll_joint": 0.28,
  "left_elbow_joint": 1.15,
  "right_shoulder_pitch_joint": 0.75,
  "right_shoulder_roll_joint": -0.28,
  "right_elbow_joint": 1.15,
}

SIDE_ROLL = {
  "left_hip_pitch_joint": -0.95,
  "left_hip_roll_joint": 0.35,
  "left_knee_joint": 1.6,
  "left_ankle_pitch_joint": -0.7,
  "right_hip_pitch_joint": -0.6,
  "right_hip_roll_joint": -0.1,
  "right_knee_joint": 1.2,
  "right_ankle_pitch_joint": -0.5,
  "waist_roll_joint": 0.25,
  "left_shoulder_pitch_joint": 0.55,
  "left_shoulder_roll_joint": 0.4,
  "left_elbow_joint": 1.05,
  "right_shoulder_pitch_joint": 0.25,
  "right_elbow_joint": 0.95,
}

CROUCH = {
  "left_hip_pitch_joint": -0.7,
  "left_knee_joint": 1.15,
  "left_ankle_pitch_joint": -0.55,
  "right_hip_pitch_joint": -0.7,
  "right_knee_joint": 1.15,
  "right_ankle_pitch_joint": -0.55,
  "waist_pitch_joint": 0.2,
  "left_shoulder_pitch_joint": 0.4,
  "right_shoulder_pitch_joint": 0.4,
}


def _merge_pose(*parts: dict[str, float]) -> list[float]:
  merged = dict(DEFAULT_STAND)
  for part in parts:
    merged.update(part)
  return [merged[name] for name in JOINT_ORDER]


def _lerp(a: float, b: float, t: float) -> float:
  return a * (1.0 - t) + b * t


def _euler_xyz_to_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
  cr = math.cos(roll * 0.5)
  sr = math.sin(roll * 0.5)
  cp = math.cos(pitch * 0.5)
  sp = math.sin(pitch * 0.5)
  cy = math.cos(yaw * 0.5)
  sy = math.sin(yaw * 0.5)
  qw = cr * cp * cy + sr * sp * sy
  qx = sr * cp * cy - cr * sp * sy
  qy = cr * sp * cy + sr * cp * sy
  qz = cr * cp * sy - sr * sp * cy
  return (qx, qy, qz, qw)


def _interpolate_pose(start: list[float], end: list[float], t: float) -> list[float]:
  return [_lerp(a, b, t) for a, b in zip(start, end, strict=True)]


def build_rows(frames_per_phase: int) -> list[list[float]]:
  phases = [
    {
      "base_pos": (0.0, 0.0, 0.12),
      "base_rpy": (math.pi, 0.0, 0.0),
      "joint_pos": _merge_pose(SUPINE_TUCK),
    },
    {
      "base_pos": (0.0, 0.0, 0.2),
      "base_rpy": (math.pi * 0.55, 0.0, 0.0),
      "joint_pos": _merge_pose(SIDE_ROLL),
    },
    {
      "base_pos": (0.0, 0.0, 0.45),
      "base_rpy": (0.35, 0.0, 0.0),
      "joint_pos": _merge_pose(CROUCH),
    },
    {
      "base_pos": (0.0, 0.0, 0.78),
      "base_rpy": (0.0, 0.0, 0.0),
      "joint_pos": _merge_pose(),
    },
  ]

  rows: list[list[float]] = []
  for start, end in zip(phases[:-1], phases[1:], strict=True):
    for idx in range(frames_per_phase):
      t = idx / float(frames_per_phase)
      pos = [_lerp(a, b, t) for a, b in zip(start["base_pos"], end["base_pos"], strict=True)]
      rpy = [_lerp(a, b, t) for a, b in zip(start["base_rpy"], end["base_rpy"], strict=True)]
      quat_xyzw = _euler_xyz_to_xyzw(*rpy)
      joint_pos = _interpolate_pose(start["joint_pos"], end["joint_pos"], t)
      rows.append([*pos, *quat_xyzw, *joint_pos])
  rows.append(
    [
      *phases[-1]["base_pos"],
      *_euler_xyz_to_xyzw(*phases[-1]["base_rpy"]),
      *phases[-1]["joint_pos"],
    ]
  )
  return rows


def write_csv(output: Path, frames_per_phase: int) -> Path:
  rows = build_rows(frames_per_phase)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerows(rows)
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate a small synthetic G1 get-up motion CSV.")
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("src/assets/motions/g1/getup_synthetic.csv"),
  )
  parser.add_argument("--frames-per-phase", type=int, default=60)
  args = parser.parse_args()
  output = write_csv(args.output, args.frames_per_phase)
  print(f"[INFO] Wrote synthetic get-up motion CSV: {output}")


if __name__ == "__main__":
  main()
