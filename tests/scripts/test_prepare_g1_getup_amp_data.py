from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.tasks.velocity.rl.getup_amp_data import (
  CANONICAL_G1_23DOF_JOINT_NAMES,
  prepare_amp_dataset,
  project_to_canonical_23dof,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "g1_getup_amp"


def test_prepare_fixture_dataset_emits_manifest_and_source_gate(tmp_path: Path) -> None:
  output = tmp_path / "prepared"
  result = prepare_amp_dataset(_FIXTURE_DIR, output, validate_only=True)

  assert result["manifest"]["accepted_count"] == 3
  assert result["manifest"]["rejected_count"] == 3
  assert result["source_gate"]["status"] == "GO"
  assert (output / "manifest.json").exists()
  assert (output / "source_gate.json").exists()

  manifest = json.loads((output / "manifest.json").read_text())
  accepted_names = {Path(item["path"]).name for item in manifest["accepted"]}
  assert "valid_getup_canonical.npz" in accepted_names
  assert "valid_getup_shuffled.npz" in accepted_names
  assert "valid_getup_full29_projected.npz" in accepted_names


def test_prepare_cli_validate_only(tmp_path: Path) -> None:
  output = tmp_path / "cli_prepared"
  completed = subprocess.run(
    [
      sys.executable,
      "scripts/prepare_g1_getup_amp_data.py",
      "--input",
      str(_FIXTURE_DIR),
      "--output",
      str(output),
      "--validate-only",
    ],
    check=True,
    cwd=Path(__file__).resolve().parents[2],
    text=True,
    capture_output=True,
  )

  assert '"status": "GO"' in completed.stdout
  assert (output / "manifest.json").exists()


def test_projection_reorders_shuffled_joints_by_name() -> None:
  payload = np.load(_FIXTURE_DIR / "valid_getup_shuffled.npz", allow_pickle=False)
  projected = project_to_canonical_23dof(payload["joint_pos"], payload["joint_vel"], payload["joint_names"])
  canonical = np.load(_FIXTURE_DIR / "valid_getup_canonical.npz", allow_pickle=False)

  assert tuple(projected.canonical_joint_names) == CANONICAL_G1_23DOF_JOINT_NAMES
  np.testing.assert_allclose(projected.joint_pos, canonical["joint_pos"])
  np.testing.assert_allclose(projected.joint_vel, canonical["joint_vel"])


def test_projection_fails_closed_for_shape_only_data() -> None:
  payload = np.load(_FIXTURE_DIR / "valid_getup_canonical.npz", allow_pickle=False)
  with pytest.raises(ValueError, match="joint_names metadata is required"):
    project_to_canonical_23dof(payload["joint_pos"], payload["joint_vel"], None)


def test_projection_accepts_known_29dof_extras_only() -> None:
  payload = np.load(_FIXTURE_DIR / "valid_getup_full29_projected.npz", allow_pickle=False)
  projected = project_to_canonical_23dof(payload["joint_pos"], payload["joint_vel"], payload["joint_names"])

  assert projected.projection["source_joint_count"] == 29
  assert set(projected.projection["dropped_extra_joints"])
