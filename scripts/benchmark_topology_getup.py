from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from topology_getup_harness import (
  build_smoke_command,
  aggregate_scenario_results,
  default_experiment_suite,
  default_robot_trial_matrix,
  default_scenarios_for_task,
  json_ready,
  normalize_stage_name,
  parse_training_health,
  evaluate_baseline_margin,
  write_json,
)


def _print_json(payload: Any, *, output: Path | None = None) -> None:
  payload = json_ready(payload)
  if output is not None:
    write_json(output, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Compatibility CLI for topology get-up benchmark workflows.",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  normalize_parser = subparsers.add_parser("normalize-stage")
  normalize_parser.add_argument("task_id")

  scenarios_parser = subparsers.add_parser("scenarios")
  scenarios_parser.add_argument("task_id")
  scenarios_parser.add_argument("--output", type=Path)

  smoke_parser = subparsers.add_parser("smoke-command")
  smoke_parser.add_argument("task_id")
  smoke_parser.add_argument("--seed", type=int, default=1)
  smoke_parser.add_argument("--iterations", type=int, default=1)
  smoke_parser.add_argument("--num-envs", type=int, default=1)
  smoke_parser.add_argument("--save-interval", type=int, default=1)
  smoke_parser.add_argument("--extra-arg", action="append", default=[])
  smoke_parser.add_argument("--output", type=Path)

  suite_parser = subparsers.add_parser("suite-plan")
  suite_parser.add_argument("--teacher-checkpoint", default="path/to/teacher.pt")
  suite_parser.add_argument("--teacher-run-dir", default=None)
  suite_parser.add_argument("--iterations", type=int, default=5000)
  suite_parser.add_argument("--num-envs", type=int, default=4096)
  suite_parser.add_argument("--output", type=Path)

  checklist_parser = subparsers.add_parser("robot-checklist")
  checklist_parser.add_argument("--output", type=Path)

  aggregate_parser = subparsers.add_parser("aggregate-summary")
  aggregate_parser.add_argument("results_json", type=Path)
  aggregate_parser.add_argument("--output", type=Path)

  compare_parser = subparsers.add_parser("compare-summary")
  compare_parser.add_argument("main_summary", type=Path)
  compare_parser.add_argument("baseline_summary", type=Path)
  compare_parser.add_argument("--per-bucket-margin", type=float, default=0.10)
  compare_parser.add_argument("--aggregate-margin", type=float, default=0.05)
  compare_parser.add_argument("--output", type=Path)

  health_parser = subparsers.add_parser("training-health")
  health_parser.add_argument("log_path", nargs="?", type=Path)
  health_parser.add_argument("--output", type=Path)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.command == "normalize-stage":
    stage_name = normalize_stage_name(args.task_id)
    if stage_name is None:
      parser.error(f"Task id does not map to a topology-getup stage: {args.task_id}")
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

  if args.command == "suite-plan":
    payload = {
      "lanes": default_experiment_suite(
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_run_dir=args.teacher_run_dir,
        iterations=args.iterations,
        num_envs=args.num_envs,
      )
    }
    _print_json(payload, output=args.output)
    return 0

  if args.command == "robot-checklist":
    _print_json({"robot_trials": default_robot_trial_matrix()}, output=args.output)
    return 0

  if args.command == "aggregate-summary":
    payload = aggregate_scenario_results(json.loads(args.results_json.read_text()))
    _print_json(payload, output=args.output)
    return 0

  if args.command == "compare-summary":
    payload = evaluate_baseline_margin(
      json.loads(args.main_summary.read_text()),
      json.loads(args.baseline_summary.read_text()),
      per_bucket_margin=args.per_bucket_margin,
      aggregate_margin=args.aggregate_margin,
    )
    _print_json(payload, output=args.output)
    return 0

  if args.command == "training-health":
    if args.log_path is None:
      import sys
      log_text = sys.stdin.read()
    else:
      log_text = args.log_path.read_text()
    _print_json(parse_training_health(log_text), output=args.output)
    return 0

  parser.error(f"Unsupported command: {args.command}")
  return 2


if __name__ == "__main__":
  raise SystemExit(main())
