from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

DEFAULT_DESTINATION_ROOT = Path("deploy/robots/g1_getup/config/policy/topology_getup/v0")
MANIFEST_NAME = "topology_getup_artifacts.json"
DEPLOYABLE_STUDENT_LANES = {"main", "naive_depth", "distill"}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Promote a deployable topology-getup student artifact bundle from a run directory "
      "into the dedicated g1_getup runtime staging layout."
    ),
  )
  parser.add_argument(
    "--run-dir",
    type=Path,
    required=True,
    help="Run directory containing topology_getup_artifacts.json and exported artifacts.",
  )
  parser.add_argument(
    "--destination-root",
    type=Path,
    default=DEFAULT_DESTINATION_ROOT,
    help="Destination staging root under deploy/robots/g1_getup.",
  )
  return parser


def _load_manifest(run_dir: Path) -> dict[str, object]:
  manifest_path = run_dir / MANIFEST_NAME
  if not manifest_path.exists():
    raise FileNotFoundError(f"Topology-getup artifact manifest not found: {manifest_path}")
  payload = json.loads(manifest_path.read_text())
  if payload.get("schema_version") != "topology_getup_artifacts_v1":
    raise ValueError(f"Unsupported topology-getup artifact manifest schema: {payload.get('schema_version')}")
  return payload


def _copy(src: Path, dst: Path) -> None:
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)


def promote_topology_getup_artifact(*, run_dir: Path, destination_root: Path) -> Path:
  source_root = run_dir.expanduser()
  if not source_root.exists():
    raise FileNotFoundError(f"Run directory not found: {source_root}")
  if not source_root.is_dir():
    raise FileNotFoundError(f"Run directory path is not a directory: {source_root}")

  payload = _load_manifest(source_root)
  lane = str(payload.get("lane", ""))
  if lane not in DEPLOYABLE_STUDENT_LANES:
    raise ValueError(f"Lane '{lane}' is not a deployable student lane.")

  policy_onnx = source_root / str(payload["policy_onnx"])
  policy_analysis_onnx = source_root / str(payload["policy_analysis_onnx"])
  deploy_yaml = source_root / str(payload["deploy_yaml"])
  for path in (policy_onnx, policy_analysis_onnx, deploy_yaml):
    if not path.exists():
      raise FileNotFoundError(f"Expected artifact file not found: {path}")

  destination = destination_root.expanduser()
  _copy(policy_onnx, destination / "exported" / "policy.onnx")
  _copy(policy_analysis_onnx, destination / "exported" / "policy_analysis.onnx")
  _copy(deploy_yaml, destination / "params" / "deploy.yaml")
  staged_manifest = dict(payload)
  staged_manifest.pop("checkpoint", None)
  staged_manifest["policy_onnx"] = "exported/policy.onnx"
  staged_manifest["policy_analysis_onnx"] = "exported/policy_analysis.onnx"
  staged_manifest["deploy_yaml"] = "params/deploy.yaml"
  staged_manifest["promoted_from_run_dir"] = str(source_root)
  if "checkpoint" in payload:
    staged_manifest["source_checkpoint"] = str(payload["checkpoint"])
  (destination / MANIFEST_NAME).write_text(json.dumps(staged_manifest, indent=2, sort_keys=True))
  return destination


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  promote_topology_getup_artifact(
    run_dir=args.run_dir,
    destination_root=args.destination_root,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
