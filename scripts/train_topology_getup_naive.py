from __future__ import annotations

import subprocess
import sys
from typing import Sequence

TASK_ID = "Unitree-G1-TopologyGetUp-Stage0-NaiveDepth"


def build_command(extra_args: Sequence[str]) -> list[str]:
  forwarded = list(extra_args)
  if forwarded and forwarded[0] == "--":
    forwarded = forwarded[1:]
  return [sys.executable, "scripts/train.py", TASK_ID, *forwarded]


def main(argv: Sequence[str] | None = None) -> int:
  command = build_command(argv or [])
  completed = subprocess.run(command, check=False)
  return completed.returncode


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
