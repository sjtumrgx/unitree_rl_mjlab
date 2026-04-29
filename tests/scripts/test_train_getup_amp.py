from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_getup_amp import AMP_TASK_ID, build_forwarded_args, validate_demo_data_dir


def test_train_getup_amp_short_args_map_to_real_tyro_overrides() -> None:
  forwarded = build_forwarded_args(
    demo_data_dir="/tmp/g1_getup_amp_fixture",
    manifest_path="/tmp/g1_getup_amp_fixture/manifest.json",
    max_iterations=1,
    num_envs=4,
    headless_smoke=True,
    extra_args=["--", "--agent.num-steps-per-env=2"],
  )

  assert forwarded[0] == AMP_TASK_ID
  assert "--agent.algorithm.demo-data-dir=/tmp/g1_getup_amp_fixture" in forwarded
  assert "--agent.algorithm.manifest-path=/tmp/g1_getup_amp_fixture/manifest.json" in forwarded
  assert "--agent.max-iterations=1" in forwarded
  assert "--env.scene.num-envs=4" in forwarded
  assert "--gpu-ids=cpu" in forwarded
  assert "--agent.logger=tensorboard" in forwarded
  assert "--agent.upload-model=False" in forwarded
  assert "--agent.num-steps-per-env=2" in forwarded


def test_train_getup_amp_missing_manifest_fails_before_training(tmp_path: Path) -> None:
  with pytest.raises(FileNotFoundError, match="prepare_g1_getup_amp_data"):
    validate_demo_data_dir(str(tmp_path / "missing"), None)


def test_train_getup_amp_rejects_missing_source_gate(tmp_path: Path) -> None:
  manifest = tmp_path / "manifest.json"
  manifest.write_text('{"accepted": []}')

  with pytest.raises(ValueError, match="source gate is missing"):
    validate_demo_data_dir(str(tmp_path), None)


def test_train_getup_amp_rejects_stop_source_gate(tmp_path: Path) -> None:
  manifest = tmp_path / "manifest.json"
  manifest.write_text('{"accepted": []}')
  (tmp_path / "source_gate.json").write_text(
    '{"status": "STOP", "stop_reasons": ["license unresolved"]}'
  )

  with pytest.raises(ValueError, match="blocks training"):
    validate_demo_data_dir(str(tmp_path), None)
