from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from antifall_harness import (
  build_smoke_command,
  default_scenarios_for_task,
  normalize_stage_name,
  parse_training_health,
  write_json,
)


def _print_json(payload: Any, *, output: Path | None = None) -> None:
  def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
      return {k: _json_ready(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
      return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
      return [_json_ready(v) for v in value]
    return value

  payload = _json_ready(payload)
  if output is not None:
    write_json(output, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Compatibility CLI for the planned benchmark_antifall interface.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  normalize_parser = subparsers.add_parser(
    "normalize-stage",
    help="Normalize an AntiFall task id to its lowercase stage name.",
  )
  normalize_parser.add_argument("task_id")

  scenarios_parser = subparsers.add_parser(
    "scenarios",
    help="Emit the default anti-fall benchmark scenarios for a task id.",
  )
  scenarios_parser.add_argument("task_id")
  scenarios_parser.add_argument(
    "--output",
    type=Path,
    help="Optional JSON file path to write alongside stdout.",
  )

  smoke_parser = subparsers.add_parser(
    "smoke-command",
    help="Build the canonical smoke-training command for an anti-fall task.",
  )
  smoke_parser.add_argument("task_id")
  smoke_parser.add_argument("--seed", type=int, default=1)
  smoke_parser.add_argument("--iterations", type=int, default=1)
  smoke_parser.add_argument("--num-envs", type=int, default=1)
  smoke_parser.add_argument("--save-interval", type=int, default=1)
  smoke_parser.add_argument(
    "--extra-arg",
    action="append",
    default=[],
    help="Extra argument appended to the generated train command.",
  )
  smoke_parser.add_argument(
    "--output",
    type=Path,
    help="Optional JSON file path to write alongside stdout.",
  )

  health_parser = subparsers.add_parser(
    "training-health",
    help="Parse training log text and summarize basic health flags.",
  )
  health_parser.add_argument(
    "log_path",
    nargs="?",
    type=Path,
    help="Optional path to the training log. Reads stdin when omitted.",
  )
  health_parser.add_argument(
    "--output",
    type=Path,
    help="Optional JSON file path to write alongside stdout.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.command == "normalize-stage":
    stage_name = normalize_stage_name(args.task_id)
    if stage_name is None:
      parser.error(f"Task id does not map to an anti-fall stage: {args.task_id}")
    print(stage_name)
    return 0

  if args.command == "scenarios":
    payload = {
      "task_id": args.task_id,
      "stage_name": normalize_stage_name(args.task_id),
      "scenarios": default_scenarios_for_task(args.task_id),
    }
    _print_json(payload, output=args.output)
    return 0

  if args.command == "smoke-command":
    payload = {
      "task_id": args.task_id,
      "stage_name": normalize_stage_name(args.task_id),
      "command": build_smoke_command(
        task_id=args.task_id,
        seed=args.seed,
        iterations=args.iterations,
        num_envs=args.num_envs,
        save_interval=args.save_interval,
        extra_args=args.extra_arg,
      ),
    }
    _print_json(payload, output=args.output)
    return 0

  if args.command == "training-health":
    if args.log_path is None:
      log_text = input() if False else None
      if log_text is None:
        import sys

        log_text = sys.stdin.read()
    else:
      log_text = args.log_path.read_text()
    payload = parse_training_health(log_text)
    _print_json(payload, output=args.output)
    return 0

  parser.error(f"Unsupported command: {args.command}")
  return 2


if __name__ == "__main__":
  raise SystemExit(main())
