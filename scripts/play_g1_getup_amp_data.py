"""Prepare and replay selected G1 GetUp AMP retargeted motion clips.

The script is intentionally data-first: it validates the selected OpenHE/LAFAN1
`.pkl` clips, writes the same AMP manifest used by training, and can replay the
root pose + 23 DoF trajectory directly on the checked-in G1 MuJoCo model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.tasks.velocity.rl.getup_amp_data import (  # noqa: E402
  CANONICAL_G1_23DOF_JOINT_NAMES,
  prepare_amp_sequences,
)

DEFAULT_LAFAN1_GETUP_FILES = (
  Path(
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/"
    "lafan1_retargeted/fallAndGetUp1_subject1.pkl"
  ),
  Path(
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/"
    "lafan1_retargeted/fallAndGetUp1_subject4.pkl"
  ),
  Path(
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/"
    "lafan1_retargeted/fallAndGetUp1_subject5.pkl"
  ),
  Path(
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/"
    "lafan1_retargeted/fallAndGetUp2_subject2.pkl"
  ),
  Path(
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/"
    "lafan1_retargeted/fallAndGetUp2_subject3.pkl"
  ),
  Path(
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/"
    "lafan1_retargeted/fallAndGetUp3_subject1.pkl"
  ),
)

DEFAULT_XML = Path("src/assets/robots/unitree_g1/xmls/g1_23dof.xml")
DEFAULT_PREPARE_OUTPUT = Path("~/unitree_rl_mjlab/data/motions/g1_getup_amp")
DEFAULT_SOURCE_URL = "https://huggingface.co/datasets/openhe/g1-retargeted-motions"
DEFAULT_SOURCE_LICENSE = "MIT"
DEFAULT_UPSTREAM_LICENSE = "LAFAN1 original source restrictions reviewed"


@dataclass(frozen=True)
class PlaybackClip:
  path: Path
  fps: float
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray
  joint_pos: np.ndarray

  @property
  def frame_count(self) -> int:
    return int(self.joint_pos.shape[0])

  @property
  def duration_s(self) -> float:
    return self.frame_count / self.fps


def _repo_path(path: Path) -> Path:
  path = path.expanduser()
  return path if path.is_absolute() else (_REPO_ROOT / path)


def resolve_motion_files(motion_files: Sequence[Path] | None) -> list[Path]:
  selected = list(motion_files or DEFAULT_LAFAN1_GETUP_FILES)
  resolved = [_repo_path(Path(path)) for path in selected]
  missing = [str(path) for path in resolved if not path.exists()]
  if missing:
    raise FileNotFoundError(f"Selected motion file(s) do not exist: {missing}")
  return resolved


def prepare_selected_motions(
  motion_files: Sequence[Path],
  output_dir: Path,
  *,
  source_revision: str | None,
  validate_only: bool = False,
  source_url: str = DEFAULT_SOURCE_URL,
  source_license: str = DEFAULT_SOURCE_LICENSE,
  upstream_license: str = DEFAULT_UPSTREAM_LICENSE,
) -> dict[str, Any]:
  """Prepare exactly the selected files into the AMP training manifest."""
  motion_files = [Path(path) for path in motion_files]
  return prepare_amp_sequences(
    motion_files,
    output_dir,
    input_label="selected:" + ",".join(str(path) for path in motion_files),
    validate_only=validate_only,
    source_url=source_url,
    source_revision=source_revision,
    source_license=source_license,
    upstream_license=upstream_license,
  )


def load_playback_clip(standardized_npz: Path) -> PlaybackClip:
  payload = np.load(standardized_npz, allow_pickle=False)
  required = {"joint_pos", "root_pos_w", "root_quat_w", "fps"}
  missing = sorted(required - set(payload.files))
  if missing:
    raise ValueError(f"{standardized_npz} is missing playback fields: {missing}")
  fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
  return PlaybackClip(
    path=standardized_npz,
    fps=fps,
    root_pos_w=np.asarray(payload["root_pos_w"], dtype=np.float64),
    root_quat_w=np.asarray(payload["root_quat_w"], dtype=np.float64),
    joint_pos=np.asarray(payload["joint_pos"], dtype=np.float64),
  )


def _qpos_addresses(model, joint_names: Sequence[str]) -> list[int]:
  addresses: list[int] = []
  for name in joint_names:
    try:
      addresses.append(int(model.joint(name).qposadr[0]))
    except Exception as exc:  # pragma: no cover - depends on MuJoCo model internals
      raise ValueError(f"MuJoCo model is missing expected G1 joint {name!r}") from exc
  return addresses


def set_mujoco_frame(
  model,
  data,
  qpos_addresses: Sequence[int],
  clip: PlaybackClip,
  frame_idx: int,
) -> None:
  if model.nq < 7 + len(qpos_addresses):
    raise ValueError(
      f"MuJoCo model nq={model.nq} cannot hold freejoint + 23 DoF playback"
    )
  frame_idx = int(np.clip(frame_idx, 0, clip.frame_count - 1))
  data.qpos[:3] = clip.root_pos_w[frame_idx]
  data.qpos[3:7] = clip.root_quat_w[frame_idx]
  for joint_index, qpos_address in enumerate(qpos_addresses):
    data.qpos[qpos_address] = clip.joint_pos[frame_idx, joint_index]


def validate_clip_headless(clip: PlaybackClip, xml_path: Path) -> dict[str, Any]:
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  qpos_addresses = _qpos_addresses(model, CANONICAL_G1_23DOF_JOINT_NAMES)
  for frame_idx in range(clip.frame_count):
    set_mujoco_frame(model, data, qpos_addresses, clip, frame_idx)
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.xpos).all():
      raise ValueError(f"Non-finite MuJoCo state at frame {frame_idx} in {clip.path}")
  return summarize_clip(clip)


def summarize_clip(clip: PlaybackClip) -> dict[str, Any]:
  root_z = clip.root_pos_w[:, 2]
  return {
    "path": str(clip.path),
    "frames": clip.frame_count,
    "fps": clip.fps,
    "duration_s": round(clip.duration_s, 3),
    "root_z_min": round(float(np.nanmin(root_z)), 4),
    "root_z_max": round(float(np.nanmax(root_z)), 4),
    "joint_abs_max": round(float(np.nanmax(np.abs(clip.joint_pos))), 4),
  }


def play_clip(
  clip: PlaybackClip,
  xml_path: Path,
  *,
  speed: float = 1.0,
  loop: bool = False,
) -> None:
  import mujoco
  import mujoco.viewer

  if speed <= 0.0:
    raise ValueError("--speed must be positive")

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  data = mujoco.MjData(model)
  qpos_addresses = _qpos_addresses(model, CANONICAL_G1_23DOF_JOINT_NAMES)
  frame_dt = 1.0 / (clip.fps * speed)
  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      for frame_idx in range(clip.frame_count):
        if not viewer.is_running():
          return
        started = time.monotonic()
        set_mujoco_frame(model, data, qpos_addresses, clip, frame_idx)
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(max(0.0, frame_dt - (time.monotonic() - started)))
      if not loop:
        return


def _accepted_output_paths(result: dict[str, Any]) -> list[Path]:
  return [
    Path(item["output_path"])
    for item in result["manifest"]["accepted"]
    if item.get("output_path")
  ]


def _train_command(output_dir: Path) -> str:
  return (
    "python scripts/train_getup_amp.py \\\n"
    f"  --demo-data-dir {output_dir.as_posix()} \\\n"
    "  --num-envs 4096 \\\n"
    "  --max-iterations 10001"
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--motion-file",
    action="append",
    type=Path,
    default=None,
    help=(
      "OpenHE/LAFAN1 .pkl to validate/play. Repeatable. "
      "Defaults to the six fallAndGetUp clips."
    ),
  )
  parser.add_argument(
    "--prepare-output",
    type=Path,
    default=DEFAULT_PREPARE_OUTPUT,
    help="AMP output directory that receives manifest.json, source_gate.json, and motions/*.npz.",
  )
  parser.add_argument(
    "--source-revision",
    default=None,
    help="Dataset commit/snapshot id recorded in source_gate.json.",
  )
  parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
  parser.add_argument("--source-license", default=DEFAULT_SOURCE_LICENSE)
  parser.add_argument("--upstream-license", default=DEFAULT_UPSTREAM_LICENSE)
  parser.add_argument(
    "--require-go",
    action="store_true",
    help="Exit non-zero unless source_gate.status is GO.",
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Prepare and headless-validate without opening a viewer.",
  )
  parser.add_argument(
    "--play-all",
    action="store_true",
    help="Replay every accepted clip instead of only --motion-index.",
  )
  parser.add_argument(
    "--motion-index",
    type=int,
    default=0,
    help="Accepted clip index to replay when --play-all is not set.",
  )
  parser.add_argument(
    "--xml",
    type=Path,
    default=DEFAULT_XML,
    help="G1 23DoF MuJoCo XML used for kinematic playback.",
  )
  parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
  parser.add_argument(
    "--loop",
    action="store_true",
    help="Loop the selected clip(s) until the viewer is closed.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  motion_files = resolve_motion_files(args.motion_file)
  output_dir = _repo_path(args.prepare_output)
  xml_path = _repo_path(args.xml)
  if not xml_path.exists():
    raise FileNotFoundError(f"MuJoCo XML does not exist: {xml_path}")

  result = prepare_selected_motions(
    motion_files,
    output_dir,
    source_revision=args.source_revision,
    validate_only=args.validate_only,
    source_url=args.source_url,
    source_license=args.source_license,
    upstream_license=args.upstream_license,
  )
  print(json.dumps(result["source_gate"], indent=2, sort_keys=True))
  accepted_paths = _accepted_output_paths(result)
  if not accepted_paths:
    print(
      "[ERROR] No selected motion clips were accepted; inspect manifest.json for rejected reasons.",
      file=sys.stderr,
    )
    return 2
  if args.require_go and result["source_gate"]["status"] != "GO":
    print(
      "[ERROR] source_gate.status is not GO; pass --source-revision and review license metadata.",
      file=sys.stderr,
    )
    return 2

  summaries = [
    validate_clip_headless(load_playback_clip(path), xml_path)
    for path in accepted_paths
  ]
  print(json.dumps({"accepted_playback": summaries}, indent=2, sort_keys=True))
  print("\nTraining command after source_gate.status is GO:")
  print(_train_command(args.prepare_output))

  if args.validate_only:
    return 0

  if not args.play_all and not 0 <= args.motion_index < len(accepted_paths):
    raise IndexError(
      f"--motion-index {args.motion_index} is out of range for {len(accepted_paths)} accepted clip(s)"
    )
  selected_paths = accepted_paths if args.play_all else [accepted_paths[args.motion_index]]
  for standardized_path in selected_paths:
    clip = load_playback_clip(standardized_path)
    print(
      f"[INFO] Playing {standardized_path} "
      f"({clip.frame_count} frames @ {clip.fps:g} FPS)"
    )
    play_clip(clip, xml_path, speed=args.speed, loop=args.loop)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
