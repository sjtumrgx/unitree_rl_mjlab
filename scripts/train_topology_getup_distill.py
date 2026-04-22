from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.tasks.velocity.rl.topology_getup_artifacts import (
  resolve_teacher_checkpoint as resolve_teacher_checkpoint_from_artifacts,
)

TASK_ID = "Unitree-G1-TopologyGetUp-Stage0-Distill"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Launch topology-getup teacher-student distillation training. "
      "This wrapper preserves existing tasks and forwards to scripts/train.py with "
      "the dedicated distillation task id."
    ),
  )
  teacher_source = parser.add_mutually_exclusive_group(required=True)
  teacher_source.add_argument(
    "--teacher-checkpoint",
    type=Path,
    help="Path to the PPO teacher checkpoint used to initialize distillation.",
  )
  teacher_source.add_argument(
    "--teacher-run-dir",
    type=Path,
    help="Path to a teacher run directory containing topology_getup_artifacts.json or model_*.pt.",
  )
  parser.add_argument(
    "extra_args",
    nargs=argparse.REMAINDER,
    help="Additional arguments forwarded to scripts/train.py after '--'.",
  )
  return parser


def resolve_teacher_checkpoint(
  *,
  teacher_checkpoint: Path | None,
  teacher_run_dir: Path | None,
) -> Path:
  return resolve_teacher_checkpoint_from_artifacts(
    teacher_checkpoint=teacher_checkpoint,
    teacher_run_dir=teacher_run_dir,
  )


def build_command(*, teacher_checkpoint: Path, extra_args: Sequence[str]) -> list[str]:
  forwarded = list(extra_args)
  if forwarded and forwarded[0] == "--":
    forwarded = forwarded[1:]
  return [
    sys.executable,
    "scripts/train.py",
    TASK_ID,
    f"--agent.teacher-load-path={teacher_checkpoint}",
    *forwarded,
  ]


def run_distill_train(
  *,
  teacher_checkpoint: Path | None = None,
  teacher_run_dir: Path | None = None,
  extra_args: Sequence[str],
) -> int:
  validated_checkpoint = resolve_teacher_checkpoint(
    teacher_checkpoint=teacher_checkpoint,
    teacher_run_dir=teacher_run_dir,
  )
  command = build_command(teacher_checkpoint=validated_checkpoint, extra_args=extra_args)
  completed = subprocess.run(command, check=False)
  return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    return run_distill_train(
      teacher_checkpoint=args.teacher_checkpoint,
      teacher_run_dir=args.teacher_run_dir,
      extra_args=args.extra_args,
    )
  except (FileNotFoundError, ValueError) as exc:
    parser.exit(status=2, message=f"error: {exc}\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
