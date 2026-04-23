from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "promote_topology_getup_artifact.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_promote_topology_getup_artifact_{len(sys.modules)}"
  sys.modules.pop(unique_name, None)
  spec = importlib.util.spec_from_file_location(unique_name, SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _invoke_main(module, args: list[str]):
  main = getattr(module, "main")
  parameters = inspect.signature(main).parameters
  if len(parameters) == 0:
    old_argv = sys.argv
    sys.argv = [str(SCRIPT_PATH), *args]
    try:
      return main()
    finally:
      sys.argv = old_argv
  return main(args)


def _write_run_artifacts(run_dir: Path, *, lane: str = "distill") -> None:
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "policy.onnx").write_text("policy")
  (run_dir / "policy_analysis.onnx").write_text("analysis")
  params_dir = run_dir / "params"
  params_dir.mkdir()
  (params_dir / "deploy.yaml").write_text("support_geometry_interface: {}\n")
  (run_dir / "topology_getup_artifacts.json").write_text(
    json.dumps(
      {
        "schema_version": "topology_getup_artifacts_v1",
        "lane": lane,
        "checkpoint": "model_0.pt",
        "policy_onnx": "policy.onnx",
        "policy_analysis_onnx": "policy_analysis.onnx",
        "deploy_yaml": "params/deploy.yaml",
        "support_geometry_interface_version": "sgi_v1",
      }
    )
  )


def test_promote_run_dir_stages_bundle_into_g1_getup_layout(tmp_path: Path) -> None:
  module = _load_module()
  run_dir = tmp_path / "run"
  _write_run_artifacts(run_dir, lane="distill")
  staging_root = tmp_path / "deploy" / "robots" / "g1_getup" / "config" / "policy" / "topology_getup" / "v0"

  promoted = module.promote_topology_getup_artifact(
    run_dir=run_dir,
    destination_root=staging_root,
  )

  assert promoted == staging_root
  assert (staging_root / "exported" / "policy.onnx").read_text() == "policy"
  assert (staging_root / "exported" / "policy_analysis.onnx").read_text() == "analysis"
  assert (staging_root / "params" / "deploy.yaml").read_text() == "support_geometry_interface: {}\n"
  manifest = json.loads((staging_root / "topology_getup_artifacts.json").read_text())
  assert manifest["lane"] == "distill"
  assert manifest["policy_onnx"] == "exported/policy.onnx"
  assert manifest["policy_analysis_onnx"] == "exported/policy_analysis.onnx"
  assert manifest["deploy_yaml"] == "params/deploy.yaml"
  assert manifest["promoted_from_run_dir"] == str(run_dir)
  assert manifest["source_checkpoint"] == "model_0.pt"
  assert "checkpoint" not in manifest


def test_promote_rejects_teacher_lane_as_not_deployable(tmp_path: Path) -> None:
  module = _load_module()
  run_dir = tmp_path / "teacher_run"
  _write_run_artifacts(run_dir, lane="teacher")

  with pytest.raises(ValueError, match="not a deployable student lane"):
    module.promote_topology_getup_artifact(
      run_dir=run_dir,
      destination_root=tmp_path / "staging",
    )


def test_main_promotes_from_cli_args(tmp_path: Path) -> None:
  module = _load_module()
  run_dir = tmp_path / "run"
  _write_run_artifacts(run_dir, lane="main")
  staging_root = tmp_path / "staging"

  result = _invoke_main(
    module,
    [
      "--run-dir",
      str(run_dir),
      "--destination-root",
      str(staging_root),
    ],
  )
  assert result == 0
  assert (staging_root / "exported" / "policy.onnx").exists()


def test_promote_allows_student_lane_without_analysis_export(tmp_path: Path) -> None:
  module = _load_module()
  run_dir = tmp_path / "naive_run"
  _write_run_artifacts(run_dir, lane="naive_depth")
  (run_dir / "policy_analysis.onnx").unlink()
  manifest = json.loads((run_dir / "topology_getup_artifacts.json").read_text())
  manifest.pop("policy_analysis_onnx", None)
  (run_dir / "topology_getup_artifacts.json").write_text(json.dumps(manifest))

  staging_root = tmp_path / "staging"
  promoted = module.promote_topology_getup_artifact(
    run_dir=run_dir,
    destination_root=staging_root,
  )

  assert promoted == staging_root
  assert (staging_root / "exported" / "policy.onnx").read_text() == "policy"
  assert not (staging_root / "exported" / "policy_analysis.onnx").exists()
  staged_manifest = json.loads((staging_root / "topology_getup_artifacts.json").read_text())
  assert staged_manifest["lane"] == "naive_depth"
  assert "policy_analysis_onnx" not in staged_manifest
