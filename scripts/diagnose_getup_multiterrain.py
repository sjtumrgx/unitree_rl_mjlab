"""Evaluate one Unitree-G1-GetUp checkpoint across all HoST terrain variants.

This is a thin orchestration wrapper around ``diagnose_getup_rollout.py``.  It
keeps the acceptance contract explicit: the same checkpoint must satisfy the
success threshold on every requested terrain, and per-terrain risk flags are
reported rather than hidden behind a single aggregate pass/fail bit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts import diagnose_getup_rollout as rollout
from src.tasks.velocity.config.g1_getup.env_cfgs import GETUP_SINGLE_TERRAIN_VARIANTS

SCHEMA_VERSION = "g1-getup-multiterrain-v1"


def build_rollout_args(
  *,
  terrain: str,
  checkpoint_file: Path | str | None,
  agent: str,
  num_envs: int,
  steps: int,
  device: str,
  stop_on_done: bool,
  train_like: bool = False,
  output: Path | None = None,
) -> argparse.Namespace:
  return argparse.Namespace(
    task_id="Unitree-G1-GetUp",
    getup_terrain=terrain,
    demo_data_dir="data/motions/g1_getup_amp",
    manifest_path=None,
    checkpoint_file=str(checkpoint_file) if checkpoint_file is not None else None,
    agent=agent,
    num_envs=int(num_envs),
    steps=int(steps),
    device=device,
    train_like=bool(train_like),
    stop_on_done=bool(stop_on_done),
    output=output,
  )


def _success_rate(summary: dict[str, Any]) -> float | None:
  try:
    value = summary.get("success", {}).get("single_episode_success_rate")
  except AttributeError:
    return None
  return float(value) if value is not None else None


def _success_count(summary: dict[str, Any]) -> int | None:
  try:
    value = summary.get("success", {}).get("success_count_estimate")
  except AttributeError:
    return None
  return int(value) if value is not None else None


def _risk_flags(summary: dict[str, Any]) -> dict[str, bool]:
  flags = summary.get("risk_flags", {}) if isinstance(summary, dict) else {}
  return {str(key): bool(value) for key, value in flags.items()}


def aggregate_terrain_summaries(
  *,
  checkpoint_file: str | Path | None,
  terrains: dict[str, dict[str, Any]],
  success_threshold: float,
) -> dict[str, Any]:
  terrain_success_rates = {
    terrain: _success_rate(summary) for terrain, summary in terrains.items()
  }
  terrain_success_counts = {
    terrain: _success_count(summary) for terrain, summary in terrains.items()
  }
  terrain_gate = {
    terrain: rate is not None and rate >= float(success_threshold)
    for terrain, rate in terrain_success_rates.items()
  }
  combined_risk_flags: dict[str, bool] = {}
  for summary in terrains.values():
    for key, value in _risk_flags(summary).items():
      combined_risk_flags[key] = bool(combined_risk_flags.get(key, False) or value)

  weakest_terrain = None
  finite_rates = {
    terrain: rate for terrain, rate in terrain_success_rates.items() if rate is not None
  }
  if finite_rates:
    weakest_terrain = min(finite_rates, key=finite_rates.__getitem__)

  return {
    "schema_version": SCHEMA_VERSION,
    "type": "summary",
    "status": "ok",
    "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else None,
    "terrains": terrains,
    "success_threshold": float(success_threshold),
    "terrain_success_rates": terrain_success_rates,
    "terrain_success_counts": terrain_success_counts,
    "terrain_gate": terrain_gate,
    "all_terrains_success": bool(terrain_gate) and all(terrain_gate.values()),
    "weakest_terrain": weakest_terrain,
    "combined_risk_flags": combined_risk_flags,
  }


def run_multiterrain(args: argparse.Namespace) -> dict[str, Any]:
  terrain_summaries: dict[str, dict[str, Any]] = {}
  for terrain in args.terrains:
    rollout_args = build_rollout_args(
      terrain=terrain,
      checkpoint_file=args.checkpoint_file,
      agent=args.agent,
      num_envs=args.num_envs,
      steps=args.steps,
      device=args.device,
      stop_on_done=args.stop_on_done,
      train_like=args.train_like,
    )
    records = rollout._run_rollout_records(rollout_args)
    terrain_summaries[terrain] = records[-1]
  return aggregate_terrain_summaries(
    checkpoint_file=args.checkpoint_file,
    terrains=terrain_summaries,
    success_threshold=args.success_threshold,
  )


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", required=True, type=Path)
  parser.add_argument(
    "--terrains",
    nargs="+",
    default=list(GETUP_SINGLE_TERRAIN_VARIANTS),
    choices=GETUP_SINGLE_TERRAIN_VARIANTS,
  )
  parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=700)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--train-like", action="store_true")
  parser.add_argument("--stop-on-done", action="store_true")
  parser.add_argument("--success-threshold", type=float, default=0.95)
  parser.add_argument("--output", type=Path, default=None)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    with contextlib.redirect_stdout(sys.stderr):
      report = run_multiterrain(args)
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["all_terrains_success"] else 1
  except Exception as exc:
    blocker = rollout.build_blocker_record(
      task_id="Unitree-G1-GetUp",
      phase="multiterrain_rollout",
      exc=exc,
      request={
        "checkpoint_file": str(args.checkpoint_file),
        "terrains": list(args.terrains),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "device": args.device,
      },
    )
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(blocker, indent=2, sort_keys=True))
    print(json.dumps(blocker, sort_keys=True))
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
