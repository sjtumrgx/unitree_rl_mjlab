"""Validate and standardize external G1 GetUp AMP demonstration data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.tasks.velocity.rl.getup_amp_data import prepare_amp_dataset

DEFAULT_RAW_DATA_DIR = Path("~/unitree_rl_mjlab/data/g1-retargeted-motions")
DEFAULT_PREPARED_DATA_DIR = Path("~/unitree_rl_mjlab/data/motions/g1_getup_amp")
DEFAULT_SOURCE_URL = "https://huggingface.co/datasets/openhe/g1-retargeted-motions"
DEFAULT_SOURCE_LICENSE = "MIT"
DEFAULT_UPSTREAM_LICENSE = "LAFAN1 original source restrictions reviewed"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    default=DEFAULT_RAW_DATA_DIR,
    type=Path,
    help=(
      "Raw NPZ/PKL file or directory. Defaults to "
      f"{DEFAULT_RAW_DATA_DIR.as_posix()}."
    ),
  )
  parser.add_argument(
    "--output",
    default=DEFAULT_PREPARED_DATA_DIR,
    type=Path,
    help=(
      "Prepared output directory. Defaults to "
      f"{DEFAULT_PREPARED_DATA_DIR.as_posix()}."
    ),
  )
  parser.add_argument(
    "--validate-only",
    "--validate",
    action="store_true",
    help="Run validation/preparation without claiming real-data training readiness.",
  )
  parser.add_argument(
    "--source-url",
    default=DEFAULT_SOURCE_URL,
    help="Dataset URL recorded in source_gate.json.",
  )
  parser.add_argument(
    "--source-revision",
    default=None,
    help="Dataset revision/commit recorded in source_gate.json.",
  )
  parser.add_argument(
    "--source-license",
    default=DEFAULT_SOURCE_LICENSE,
    help="Dataset-host license recorded in source_gate.json.",
  )
  parser.add_argument(
    "--upstream-license",
    default=DEFAULT_UPSTREAM_LICENSE,
    help="Upstream source/subset license restrictions recorded in source_gate.json.",
  )
  parser.add_argument(
    "--require-go",
    action="store_true",
    help="Exit non-zero if source_gate.json status is STOP.",
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  result = prepare_amp_dataset(
    input_dir=args.input.expanduser(),
    output_dir=args.output.expanduser(),
    validate_only=args.validate_only,
    source_url=args.source_url,
    source_revision=args.source_revision,
    source_license=args.source_license,
    upstream_license=args.upstream_license,
  )
  print(json.dumps(result["source_gate"], indent=2, sort_keys=True))
  if args.require_go and result["source_gate"]["status"] != "GO":
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
