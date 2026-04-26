#!/usr/bin/env python3
"""Compare Python play and C++/DDS G1 parkour gait JSONL captures.

The comparison is intentionally index-based rather than trying to synchronize
wall-clock time.  Both capture paths run the same exported policy at 50 Hz in
flat fixed-command diagnostics; early divergence is therefore useful evidence:
observation/action mismatches show up in actor outputs, while low-level control
mismatches show up after similar targets produce different measured joints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_FIELDS = (
  "command",
  "base_ang_vel",
  "projected_gravity",
  "raw_action_policy_order",
  "applied_action_deploy_order",
  "target_q_deploy_order",
  "joint_pos_deploy_order",
  "joint_vel_deploy_order",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--python-jsonl", type=Path, required=True)
  parser.add_argument("--cpp-jsonl", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--skip-samples", type=int, default=0)
  parser.add_argument("--max-samples", type=int, default=500)
  parser.add_argument(
    "--early-samples",
    type=int,
    default=20,
    help="Also compute an early-window comparison from the start of the selected slice.",
  )
  parser.add_argument(
    "--field",
    action="append",
    dest="fields",
    help="Field to compare; may be repeated. Defaults to the standard gait parity fields.",
  )
  return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with path.expanduser().open("r", encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, start=1):
      stripped = line.strip()
      if not stripped:
        continue
      try:
        rows.append(json.loads(stripped))
      except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
  if not rows:
    raise ValueError(f"No gait samples found in {path}")
  return rows


def _as_float_list(value: Any) -> list[float] | None:
  if value is None:
    return None
  if isinstance(value, dict):
    # Stats dictionaries are intentionally not treated as comparable vectors.
    return None
  if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
    return None
  values: list[float] = []
  for item in value:
    try:
      values.append(float(item))
    except (TypeError, ValueError):
      return None
  return values


def _rmse(values: list[float]) -> float:
  if not values:
    return 0.0
  return math.sqrt(sum(value * value for value in values) / len(values))


def _mean_abs(values: list[float]) -> float:
  if not values:
    return 0.0
  return sum(abs(value) for value in values) / len(values)


def _max_abs(values: list[float]) -> float:
  return max((abs(value) for value in values), default=0.0)


def _compare_field(
  python_rows: list[dict[str, Any]],
  cpp_rows: list[dict[str, Any]],
  field: str,
) -> dict[str, Any]:
  diffs: list[float] = []
  sample_count = 0
  dim: int | None = None
  first_python: list[float] | None = None
  first_cpp: list[float] | None = None
  for py_row, cpp_row in zip(python_rows, cpp_rows, strict=False):
    py_values = _as_float_list(py_row.get(field))
    cpp_values = _as_float_list(cpp_row.get(field))
    if py_values is None or cpp_values is None or len(py_values) != len(cpp_values):
      continue
    if first_python is None:
      first_python = py_values[:8]
      first_cpp = cpp_values[:8]
    dim = len(py_values)
    diffs.extend(py - cpp for py, cpp in zip(py_values, cpp_values, strict=True))
    sample_count += 1
  return {
    "samples": sample_count,
    "dim": dim,
    "mae": _mean_abs(diffs),
    "rmse": _rmse(diffs),
    "max_abs": _max_abs(diffs),
    "first_python_head": first_python,
    "first_cpp_head": first_cpp,
  }


def _diagnose(field_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
  action_mae = float(field_stats.get("raw_action_policy_order", {}).get("mae", 0.0))
  target_mae = float(field_stats.get("target_q_deploy_order", {}).get("mae", 0.0))
  joint_pos_mae = float(field_stats.get("joint_pos_deploy_order", {}).get("mae", 0.0))
  gravity_mae = float(field_stats.get("projected_gravity", {}).get("mae", 0.0))
  command_mae = float(field_stats.get("command", {}).get("mae", 0.0))

  findings: list[str] = []
  likely_source = "undetermined"
  if command_mae > 1.0e-3:
    findings.append("velocity commands differ; compare runs are not using the same command input")
    likely_source = "command setup"
  if gravity_mae > 0.05:
    findings.append("projected gravity differs early; observation frame/state alignment is suspect")
    likely_source = "observation frame/state"
  if action_mae > 0.1:
    findings.append("actor outputs differ; inspect proprio/depth history before low-level control")
    likely_source = "policy input or depth/proprio history"
  elif target_mae > 0.05:
    findings.append("actor outputs are close but target q differs; action scaling/order is suspect")
    likely_source = "action scaling/order"
  elif joint_pos_mae > 0.05:
    findings.append("targets are close but measured joints differ; low-level PD/sim actuation is suspect")
    likely_source = "low-level controller or simulator dynamics"
  else:
    findings.append("early gait vectors are close within coarse thresholds")
    likely_source = "no large early mismatch"
  return {
    "likely_source": likely_source,
    "findings": findings,
    "thresholds": {
      "command_mae": 1.0e-3,
      "projected_gravity_mae": 0.05,
      "raw_action_policy_mae": 0.1,
      "target_q_mae": 0.05,
      "joint_pos_mae": 0.05,
    },
  }


def _diagnose_with_early_window(
  *,
  full_stats: dict[str, dict[str, Any]],
  early_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  full = _diagnose(full_stats)
  early = _diagnose(early_stats)
  early_action_mae = float(early_stats.get("raw_action_policy_order", {}).get("mae", 0.0))
  early_target_mae = float(early_stats.get("target_q_deploy_order", {}).get("mae", 0.0))
  early_joint_pos_mae = float(early_stats.get("joint_pos_deploy_order", {}).get("mae", 0.0))
  full_action_mae = float(full_stats.get("raw_action_policy_order", {}).get("mae", 0.0))
  full_joint_vel_mae = float(full_stats.get("joint_vel_deploy_order", {}).get("mae", 0.0))
  if (
    early_action_mae <= 0.1
    and early_target_mae <= 0.05
    and early_joint_pos_mae <= 0.05
    and full_action_mae > 0.1
  ):
    return {
      **full,
      "likely_source": "downstream dynamics/control divergence after initially aligned policy inputs",
      "early_window_likely_source": early["likely_source"],
      "findings": [
        "early actor outputs, targets, and joint positions are aligned",
        "later actor outputs diverge after measured joint velocity/state trajectories diverge",
        f"full-window joint_vel MAE={full_joint_vel_mae:.6g}; inspect native MuJoCo/Unitree PD timing and dynamics rather than depth/model export first",
      ],
    }
  return {
    **full,
    "early_window_likely_source": early["likely_source"],
  }


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  python_rows = _load_jsonl(args.python_jsonl)
  cpp_rows = _load_jsonl(args.cpp_jsonl)
  skip = max(0, args.skip_samples)
  max_samples = max(1, args.max_samples)
  python_slice = python_rows[skip : skip + max_samples]
  cpp_slice = cpp_rows[skip : skip + max_samples]
  fields = tuple(args.fields or DEFAULT_FIELDS)
  field_stats = {
    field: _compare_field(python_slice, cpp_slice, field)
    for field in fields
  }
  early_count = max(1, min(args.early_samples, len(python_slice), len(cpp_slice)))
  early_stats = {
    field: _compare_field(python_slice[:early_count], cpp_slice[:early_count], field)
    for field in fields
  }
  sample_count = min(len(python_slice), len(cpp_slice))
  summary = {
    "status": "ok" if sample_count > 0 else "failed",
    "python_jsonl": str(args.python_jsonl),
    "cpp_jsonl": str(args.cpp_jsonl),
    "skip_samples": skip,
    "max_samples": max_samples,
    "aligned_samples": sample_count,
    "python_samples_total": len(python_rows),
    "cpp_samples_total": len(cpp_rows),
    "fields": field_stats,
    "early_samples": early_count,
    "early_fields": early_stats,
    "diagnosis": _diagnose_with_early_window(
      full_stats=field_stats,
      early_stats=early_stats,
    ),
  }
  args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
  args.output_json.expanduser().write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0 if sample_count > 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
