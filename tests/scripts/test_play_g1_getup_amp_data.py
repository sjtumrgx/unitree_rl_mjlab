from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.tasks.velocity.rl.getup_amp_data import CANONICAL_G1_23DOF_JOINT_NAMES


def _write_openhe_clip(path: Path, *, frames: int = 5, fps: int = 30) -> None:
  joint_count = len(CANONICAL_G1_23DOF_JOINT_NAMES)
  joint_pos = np.zeros((frames, joint_count), dtype=np.float32)
  root_pos = np.stack(
    [np.zeros(frames), np.zeros(frames), np.linspace(0.25, 0.8, frames)],
    axis=1,
  ).astype(np.float32)
  # OpenHE stores SciPy-style xyzw; prepared data and MuJoCo playback use wxyz.
  root_quat_xyzw = np.tile(
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    (frames, 1),
  )
  with path.open("wb") as f:
    pickle.dump(
      {
        path.stem: {
          "root_trans_offset": root_pos,
          "root_rot": root_quat_xyzw,
          "dof": joint_pos,
          "fps": fps,
        }
      },
      f,
    )


def test_default_lafan1_fall_getup_selection_is_explicit() -> None:
  from scripts.play_g1_getup_amp_data import DEFAULT_LAFAN1_GETUP_FILES

  assert tuple(path.as_posix() for path in DEFAULT_LAFAN1_GETUP_FILES) == (
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp1_subject1.pkl",
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp1_subject4.pkl",
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp1_subject5.pkl",
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp2_subject2.pkl",
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp2_subject3.pkl",
    "~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp3_subject1.pkl",
  )


def test_prepare_selected_motions_writes_only_selected_manifest(tmp_path: Path) -> None:
  from scripts.play_g1_getup_amp_data import prepare_selected_motions

  raw_dir = tmp_path / "raw"
  raw_dir.mkdir()
  selected_a = raw_dir / "fallAndGetUp1_subject1.pkl"
  selected_b = raw_dir / "fallAndGetUp2_subject2.pkl"
  ignored = raw_dir / "walk1_subject1.pkl"
  _write_openhe_clip(selected_a)
  _write_openhe_clip(selected_b)
  _write_openhe_clip(ignored)

  output = tmp_path / "prepared"
  result = prepare_selected_motions(
    [selected_a, selected_b],
    output,
    source_revision="unit-test",
  )

  assert result["source_gate"]["status"] == "GO"
  assert result["manifest"]["accepted_count"] == 2
  accepted_names = {Path(item["path"]).name for item in result["manifest"]["accepted"]}
  assert accepted_names == {selected_a.name, selected_b.name}
  assert ignored.name not in accepted_names
  standardized = Path(result["manifest"]["accepted"][0]["output_path"])
  payload = np.load(standardized, allow_pickle=False)
  np.testing.assert_allclose(
    payload["root_quat_w"][0],
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
  )


def test_cli_validate_only_prepares_selected_files_and_prints_training_hint(tmp_path: Path) -> None:
  raw_dir = tmp_path / "raw"
  raw_dir.mkdir()
  selected_a = raw_dir / "fallAndGetUp1_subject1.pkl"
  selected_b = raw_dir / "fallAndGetUp3_subject1.pkl"
  _write_openhe_clip(selected_a)
  _write_openhe_clip(selected_b)

  output = tmp_path / "prepared"
  completed = subprocess.run(
    [
      sys.executable,
      "scripts/play_g1_getup_amp_data.py",
      "--motion-file",
      str(selected_a),
      "--motion-file",
      str(selected_b),
      "--prepare-output",
      str(output),
      "--source-revision",
      "unit-test",
      "--validate-only",
    ],
    check=True,
    cwd=Path(__file__).resolve().parents[2],
    text=True,
    capture_output=True,
  )

  assert '"accepted_count": 2' in completed.stdout
  assert "python scripts/train_getup_amp.py" in completed.stdout
  manifest = json.loads((output / "manifest.json").read_text())
  assert manifest["accepted_count"] == 2
