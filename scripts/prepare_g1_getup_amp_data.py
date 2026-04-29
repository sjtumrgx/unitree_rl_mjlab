"""Validate and standardize external G1 GetUp AMP demonstration data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts.g1_getup_amp_config import (
  DEFAULT_CONFIG_PATH,
  collect_sequence_paths,
  load_workflow_config,
  path_list,
  repo_path,
  section,
  source_metadata_from_config,
)
from src.tasks.velocity.rl.getup_amp_data import prepare_amp_dataset, prepare_amp_sequences

DEFAULT_RAW_DATA_DIR = Path("~/unitree_rl_mjlab/data/g1-retargeted-motions")
DEFAULT_PREPARED_DATA_DIR = Path("~/unitree_rl_mjlab/data/motions/g1_getup_amp")
DEFAULT_SOURCE_URL = "https://huggingface.co/datasets/openhe/g1-retargeted-motions"
DEFAULT_SOURCE_LICENSE = "MIT"
DEFAULT_UPSTREAM_LICENSE = "LAFAN1 original source restrictions reviewed"


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default=DEFAULT_CONFIG_PATH,
    type=Path,
    help=(
      "YAML workflow config. Defaults to "
      f"{DEFAULT_CONFIG_PATH.as_posix()}."
    ),
  )
  parser.add_argument(
    "--input",
    action="append",
    default=None,
    type=Path,
    help=(
      "Raw NPZ/PKL file or directory. Repeatable. Overrides prepare.inputs "
      "from the YAML config."
    ),
  )
  parser.add_argument(
    "--output",
    default=None,
    type=Path,
    help=(
      "Prepared output directory. Overrides prepare.output_dir from the YAML config."
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
    default=None,
    help="Dataset URL recorded in source_gate.json.",
  )
  parser.add_argument(
    "--source-revision",
    default=None,
    help="Dataset revision/commit recorded in source_gate.json.",
  )
  parser.add_argument(
    "--source-license",
    default=None,
    help="Dataset-host license recorded in source_gate.json.",
  )
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
  config = load_workflow_config(args.config)
  prepare_cfg = section(config, "prepare")
  configured_inputs = path_list(prepare_cfg.get("inputs"))
  input_paths = args.input if args.input is not None else configured_inputs
  output_dir = (
    args.output
    if args.output is not None
    else Path(prepare_cfg.get("output_dir", DEFAULT_PREPARED_DATA_DIR))
  )
  source = source_metadata_from_config(
    config,
    default_source_url=DEFAULT_SOURCE_URL,
    default_source_license=DEFAULT_SOURCE_LICENSE,
    default_upstream_license=DEFAULT_UPSTREAM_LICENSE,
    cli_source_url=args.source_url,
    cli_source_revision=args.source_revision,
    cli_source_license=args.source_license,
    cli_upstream_license=args.upstream_license,
  )

  if input_paths:
    resolved_inputs = [repo_path(path) for path in input_paths]
    sequence_paths = collect_sequence_paths(resolved_inputs)
    result = prepare_amp_sequences(
      sequence_paths,
      repo_path(output_dir),
      input_label=[str(path) for path in resolved_inputs],
      validate_only=args.validate_only,
      source_url=source.source_url,
      source_revision=source.source_revision,
      source_license=source.source_license,
      upstream_license=source.upstream_license,
    )
  else:
    result = prepare_amp_dataset(
      input_dir=DEFAULT_RAW_DATA_DIR.expanduser(),
      output_dir=repo_path(output_dir),
      validate_only=args.validate_only,
      source_url=source.source_url,
      source_revision=source.source_revision,
      source_license=source.source_license,
      upstream_license=source.upstream_license,
    )
  print(json.dumps(result["source_gate"], indent=2, sort_keys=True))
  if args.require_go and result["source_gate"]["status"] != "GO":
    return 2
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
