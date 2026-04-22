"""Artifact helpers for topology get-up training/export lanes."""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_TOPOLOGY_GETUP_ARTIFACT_MANIFEST = "topology_getup_artifacts.json"
TOPOLOGY_GETUP_ARTIFACT_SCHEMA_VERSION = "topology_getup_artifacts_v1"

_LANE_BY_EXPERIMENT = {
  "g1_topology_getup": "main",
  "g1_topology_getup_naive": "naive_depth",
  "g1_topology_getup_teacher": "teacher",
  "g1_topology_getup_distill": "distill",
}
_MODEL_CHECKPOINT_PATTERN = re.compile(r"model_(?P<iteration>\d+)\.pt$")


def infer_topology_getup_lane(experiment_name: str) -> str:
  return _LANE_BY_EXPERIMENT.get(experiment_name, "unknown")


def _relative_to(base_dir: Path, target: Path) -> str:
  try:
    return str(target.relative_to(base_dir))
  except ValueError:
    return str(target)


def write_topology_getup_artifact_manifest(
  *,
  output_dir: str | Path,
  experiment_name: str,
  checkpoint_path: str | Path,
  support_geometry_interface_version: str,
  distillation_mode: str | None = None,
  teacher_checkpoint: str | Path | None = None,
) -> Path:
  output_dir_path = Path(output_dir)
  checkpoint_path = Path(checkpoint_path)
  payload = {
    "schema_version": TOPOLOGY_GETUP_ARTIFACT_SCHEMA_VERSION,
    "lane": infer_topology_getup_lane(experiment_name),
    "experiment_name": experiment_name,
    "checkpoint": _relative_to(output_dir_path, checkpoint_path),
    "policy_onnx": "policy.onnx",
    "policy_analysis_onnx": "policy_analysis.onnx",
    "deploy_yaml": "params/deploy.yaml",
    "support_geometry_interface_version": support_geometry_interface_version,
  }
  if distillation_mode is not None:
    payload["distillation_mode"] = distillation_mode
  if teacher_checkpoint is not None:
    payload["teacher_checkpoint"] = str(Path(teacher_checkpoint))

  manifest_path = output_dir_path / DEFAULT_TOPOLOGY_GETUP_ARTIFACT_MANIFEST
  manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
  return manifest_path


def validate_checkpoint_file(path: str | Path) -> Path:
  checkpoint_path = Path(path).expanduser()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Teacher checkpoint file not found: {checkpoint_path}")
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Teacher checkpoint path is not a file: {checkpoint_path}")
  return checkpoint_path


def resolve_teacher_checkpoint(
  *,
  teacher_checkpoint: str | Path | None,
  teacher_run_dir: str | Path | None,
) -> Path:
  if teacher_checkpoint is not None:
    return validate_checkpoint_file(teacher_checkpoint)
  if teacher_run_dir is None:
    raise ValueError("Either teacher_checkpoint or teacher_run_dir must be provided.")

  run_dir = Path(teacher_run_dir).expanduser()
  if not run_dir.exists():
    raise FileNotFoundError(f"Teacher run directory not found: {run_dir}")
  if not run_dir.is_dir():
    raise FileNotFoundError(f"Teacher run path is not a directory: {run_dir}")

  manifest_path = run_dir / DEFAULT_TOPOLOGY_GETUP_ARTIFACT_MANIFEST
  if manifest_path.exists():
    payload = json.loads(manifest_path.read_text())
    checkpoint_name = payload.get("checkpoint")
    if checkpoint_name:
      candidate = validate_checkpoint_file(run_dir / str(checkpoint_name))
      return candidate

  checkpoint_candidates = sorted(
    run_dir.glob("model_*.pt"),
    key=lambda path: int(_MODEL_CHECKPOINT_PATTERN.match(path.name).group("iteration"))  # type: ignore[union-attr]
    if _MODEL_CHECKPOINT_PATTERN.match(path.name)
    else -1,
    reverse=True,
  )
  if checkpoint_candidates:
    return validate_checkpoint_file(checkpoint_candidates[0])

  raise FileNotFoundError(
    f"No teacher checkpoint could be resolved from run directory: {run_dir}"
  )
