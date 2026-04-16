from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from scripts.play import PlayConfig, run_play

DEFAULT_TASK = "Unitree-G1-AntiFall-Stage1"
ALLOWED_TASKS = (
  "Unitree-G1-AntiFall-Stage0",
  "Unitree-G1-AntiFall-Stage1",
  "Unitree-G1-AntiFall-Stage2",
  "Unitree-G1-AntiFall-Stage3",
  "Unitree-G1-AntiFall-Stage4a",
  "Unitree-G1-AntiFall-Stage4b",
)
ALLOWED_TASKS_TEXT = ", ".join(ALLOWED_TASKS)
_ALLOWED_TASK_SET = frozenset(ALLOWED_TASKS)
_NON_STAGE_ANTIFALL_TASKS = frozenset(
  (
    "Unitree-G1-AntiFall-Benchmark",
    "Unitree-G1-AntiFall-Curriculum",
  )
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Launch a trained Unitree G1 AntiFall policy in the native MuJoCo viewer.",
  )
  parser.add_argument(
    "--checkpoint-file",
    type=Path,
    required=True,
    help="Path to a trained AntiFall checkpoint file.",
  )
  parser.add_argument(
    "--task",
    default=DEFAULT_TASK,
    help=(
      "AntiFall stage task to launch. Allowed values: "
      + ", ".join(ALLOWED_TASKS)
    ),
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    help="Override the number of play environments.",
  )
  parser.add_argument(
    "--device",
    help="Torch device override, for example cpu or cuda:0.",
  )
  return parser


def bootstrap_tasks() -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401


def validate_task(task_id: str) -> str:
  if task_id in _ALLOWED_TASK_SET:
    return task_id
  if task_id in _NON_STAGE_ANTIFALL_TASKS:
    raise ValueError(
      f"Task '{task_id}' is not supported by play_antifall; use one of the AntiFall stage tasks only."
    )
  if "AntiFall" in task_id:
    raise ValueError(
      f"Unsupported AntiFall task '{task_id}'; allowed tasks: {ALLOWED_TASKS_TEXT}"
    )
  raise ValueError(
    f"Non-AntiFall task '{task_id}' is not supported; allowed tasks: {ALLOWED_TASKS_TEXT}"
  )


def validate_checkpoint(path: Path) -> Path:
  checkpoint_path = path.expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint path is not a file: {checkpoint_path}")
  return checkpoint_path


def require_graphical_display() -> None:
  if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    return
  raise RuntimeError(
    "Native viewer requires a graphical display; DISPLAY or WAYLAND_DISPLAY must be set."
  )


def run_antifall_play(
  *,
  checkpoint_file: Path,
  task: str = DEFAULT_TASK,
  num_envs: int | None = None,
  device: str | None = None,
) -> None:
  bootstrap_tasks()
  validated_task = validate_task(task)
  validated_checkpoint = validate_checkpoint(checkpoint_file)
  require_graphical_display()

  run_play(
    task_id=validated_task,
    cfg=PlayConfig(
      agent="trained",
      checkpoint_file=str(validated_checkpoint),
      motion_file=None,
      num_envs=num_envs,
      device=device,
      video=False,
      video_length=200,
      video_height=None,
      video_width=None,
      camera=None,
      viewer="native",
      no_terminations=False,
      _demo_mode=False,
    ),
  )


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  try:
    run_antifall_play(
      checkpoint_file=args.checkpoint_file,
      task=args.task,
      num_envs=args.num_envs,
      device=args.device,
    )
  except (FileNotFoundError, RuntimeError, ValueError) as exc:
    parser.exit(status=2, message=f"error: {exc}\n")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
