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


def test_prepare_cli_reads_yaml_inputs_and_metadata(tmp_path: Path) -> None:
  import pickle

  T = 4
  joint_pos = np.zeros((T, len(CANONICAL_G1_23DOF_JOINT_NAMES)), dtype=np.float32)
  root_pos = np.stack(
    [np.zeros(T), np.zeros(T), np.linspace(0.25, 0.8, T)], axis=1
  ).astype(np.float32)
  root_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (T, 1))
  raw_dir = tmp_path / "raw"
  nested = raw_dir / "nested"
  nested.mkdir(parents=True)
  pkl_path = nested / "fallAndGetUp1_subject1.pkl"
  with pkl_path.open("wb") as f:
    pickle.dump(
      {
        pkl_path.stem: {
          "root_trans_offset": root_pos,
          "root_rot": root_quat_xyzw,
          "dof": joint_pos,
          "fps": np.array(30),
        }
      },
      f,
    )
  output = tmp_path / "prepared"
  config = tmp_path / "g1_getup_amp.yaml"
  config.write_text(
    "\n".join(
      [
        "dataset:",
        "  source_url: https://example.invalid/g1-retargeted-motions",
        "  source_revision: unit-test-snapshot",
        "  source_license: MIT",
        "  upstream_license: local review complete",
        "prepare:",
        "  inputs:",
        f"    - {raw_dir}",
        f"  output_dir: {output}",
      ]
    )
  )

  completed = subprocess.run(
    [
      sys.executable,
      "scripts/prepare_g1_getup_amp_data.py",
      "--config",
      str(config),
    ],
    check=True,
    cwd=Path(__file__).resolve().parents[2],
    text=True,
    capture_output=True,
  )

  assert '"source_revision": "unit-test-snapshot"' in completed.stdout
  assert (output / "manifest.json").exists()
  manifest = json.loads((output / "manifest.json").read_text())
  assert manifest["accepted_count"] == 1
  assert Path(manifest["accepted"][0]["path"]) == pkl_path


def test_prepare_cli_defaults_to_yaml_config_path() -> None:
  from scripts.prepare_g1_getup_amp_data import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PREPARED_DATA_DIR,
    DEFAULT_RAW_DATA_DIR,
    build_parser,
  )

  args = build_parser().parse_args([])

  assert DEFAULT_RAW_DATA_DIR == Path("~/unitree_rl_mjlab/data/g1-retargeted-motions")
  assert DEFAULT_PREPARED_DATA_DIR == Path("~/unitree_rl_mjlab/data/motions/g1_getup_amp")
  assert DEFAULT_CONFIG_PATH == Path("data/g1_getup_amp.yaml")
  assert args.config == DEFAULT_CONFIG_PATH
  assert args.input is None
  assert args.output is None


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


def test_prepare_openhe_pkl_source_with_metadata(tmp_path: Path) -> None:
  import pickle

  T = 4
  joint_pos = np.zeros((T, len(CANONICAL_G1_23DOF_JOINT_NAMES)), dtype=np.float32)
  root_pos = np.stack(
    [np.zeros(T), np.zeros(T), np.linspace(0.2, 0.7, T)], axis=1
  ).astype(np.float32)
  root_quat_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (T, 1))
  raw_dir = tmp_path / "raw"
  raw_dir.mkdir()
  pkl_path = raw_dir / "A10-_Lie_to_crouch_stageii.pkl"
  with pkl_path.open("wb") as f:
    pickle.dump(
      {
        "A10-_Lie_to_crouch_stageii.npz": {
          "root_trans_offset": root_pos,
          "root_rot": root_quat_xyzw,
          "dof": joint_pos,
          "fps": np.array(30),
        }
      },
      f,
    )

  output = tmp_path / "prepared"
  result = prepare_amp_dataset(
    raw_dir,
    output,
    source_url="https://huggingface.co/datasets/openhe/g1-retargeted-motions",
    source_revision="test-revision",
    source_license="MIT",
    upstream_license="ACCAD/LAFAN1 restrictions recorded",
  )

  assert result["manifest"]["accepted_count"] == 1
  assert result["source_gate"]["status"] == "GO"
  standardized = Path(result["manifest"]["accepted"][0]["output_path"])
  payload = np.load(standardized, allow_pickle=False)
  assert payload["joint_pos"].shape == (T, 23)
  np.testing.assert_allclose(
    payload["root_quat_w"][0],
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
  )
  assert "openhe_g1_retargeted_pkl" in str(payload["projection"])
