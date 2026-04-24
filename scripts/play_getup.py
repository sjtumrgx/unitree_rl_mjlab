"""Play Unitree-G1-GetUp with an explicit HoST terrain variant."""

from __future__ import annotations

import argparse
import sys

from src.tasks.velocity.config.g1_getup.env_cfgs import GETUP_TERRAIN_VARIANTS


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--terrain", choices=GETUP_TERRAIN_VARIANTS, default="ground")
  parser.add_argument("extra_args", nargs=argparse.REMAINDER)
  return parser


def build_forwarded_args(terrain: str, extra_args: list[str]) -> list[str]:
  if terrain not in GETUP_TERRAIN_VARIANTS:
    raise ValueError(f"Unsupported terrain: {terrain}")
  forwarded_extra = extra_args[1:] if extra_args[:1] == ["--"] else extra_args
  return ["Unitree-G1-GetUp", f"--getup-terrain={terrain}", *forwarded_extra]


def main(argv: list[str] | None = None) -> None:
  parser = build_parser()
  args = parser.parse_args(argv)
  from scripts.play import main as play_main

  sys.argv = ["scripts/play.py", *build_forwarded_args(args.terrain, args.extra_args)]
  play_main()


if __name__ == "__main__":
  main()
