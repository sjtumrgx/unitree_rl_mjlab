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


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, type=Path, help="Raw NPZ file or directory of NPZ clips.")
  parser.add_argument("--output", required=True, type=Path, help="Prepared output directory.")
  parser.add_argument(
    "--validate-only",
    "--validate",
    action="store_true",
    help="Run validation/preparation without claiming real-data training readiness.",
  )
  parser.add_argument("--source-url", default=None, help="Dataset URL recorded in source_gate.json.")
  parser.add_argument("--source-revision", default=None, help="Dataset revision/commit recorded in source_gate.json.")
  parser.add_argument("--source-license", default=None, help="Dataset-host license recorded in source_gate.json.")
  parser.add_argument(
    "--upstream-license",
    default=None,
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
    input_dir=args.input,
    output_dir=args.output,
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
