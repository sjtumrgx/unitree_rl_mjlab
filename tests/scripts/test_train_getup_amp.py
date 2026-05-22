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
    warm_start_checkpoint=None,
    reset_actor_std_on_warm_start=False,
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


def test_train_getup_amp_warm_start_checkpoint_forwards_actor_only_resume() -> None:
  forwarded = build_forwarded_args(
    demo_data_dir="data/motions/g1_getup_amp",
    manifest_path="data/motions/g1_getup_amp/selected_manifest.json",
    max_iterations=500,
    num_envs=128,
    headless_smoke=False,
    warm_start_checkpoint="logs/rsl_rl/g1_getup/good/model_1000.pt",
    reset_actor_std_on_warm_start=False,
    extra_args=["--", "--gpu-ids", "[0]"],
  )

  assert "--agent.resume=True" in forwarded
  assert "--resume-checkpoint-path=logs/rsl_rl/g1_getup/good/model_1000.pt" in forwarded
  assert "--actor-only-resume=True" in forwarded
  assert "--reset-actor-std-on-resume=True" not in forwarded
  assert "--gpu-ids" in forwarded
  assert "[0]" in forwarded


def test_train_getup_amp_can_opt_into_resetting_warm_start_std() -> None:
  forwarded = build_forwarded_args(
    demo_data_dir="data/motions/g1_getup_amp",
    manifest_path=None,
    max_iterations=1,
    num_envs=2,
    headless_smoke=False,
    warm_start_checkpoint="logs/rsl_rl/g1_getup/good/model_1000.pt",
    reset_actor_std_on_warm_start=True,
    extra_args=[],
  )

  assert "--reset-actor-std-on-resume=True" in forwarded


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


def test_train_getup_amp_reads_yaml_defaults(tmp_path: Path) -> None:
  from scripts.train_getup_amp import resolve_train_settings

  config = tmp_path / "g1_getup_amp.yaml"
  config.write_text(
    "\n".join(
      [
        "train:",
        "  demo_data_dir: /tmp/prepared_g1_getup_amp",
        "  manifest_path: /tmp/prepared_g1_getup_amp/manifest.json",
        "  npz_files:",
        "    - /tmp/prepared_g1_getup_amp/motions/fallAndGetUp1_subject1.npz",
        "  num_envs: 8",
        "  max_iterations: 12",
      ]
    )
  )

  settings = resolve_train_settings(
    config_path=config,
    demo_data_dir=None,
    manifest_path=None,
    num_envs=None,
    max_iterations=None,
  )

  assert settings.demo_data_dir == "/tmp/prepared_g1_getup_amp"
  assert settings.manifest_path == "/tmp/prepared_g1_getup_amp/manifest.json"
  assert settings.npz_files == (
    "/tmp/prepared_g1_getup_amp/motions/fallAndGetUp1_subject1.npz",
  )
  assert settings.num_envs == 8
  assert settings.max_iterations == 12


def test_train_getup_amp_cli_overrides_yaml_defaults(tmp_path: Path) -> None:
  from scripts.train_getup_amp import resolve_train_settings

  config = tmp_path / "g1_getup_amp.yaml"
  config.write_text(
    "\n".join(
      [
        "train:",
        "  demo_data_dir: /tmp/from-config",
        "  num_envs: 8",
        "  max_iterations: 12",
      ]
    )
  )

  settings = resolve_train_settings(
    config_path=config,
    demo_data_dir="/tmp/from-cli",
    manifest_path=None,
    num_envs=4,
    max_iterations=1,
  )

  assert settings.demo_data_dir == "/tmp/from-cli"
  assert settings.manifest_path is None
  assert settings.num_envs == 4
  assert settings.max_iterations == 1


def test_train_getup_amp_writes_selected_npz_manifest(tmp_path: Path) -> None:
  import json

  from scripts.train_getup_amp import write_selected_train_manifest

  demo_dir = tmp_path / "prepared"
  motions_dir = demo_dir / "motions"
  motions_dir.mkdir(parents=True)
  selected = motions_dir / "fallAndGetUp1_subject1.npz"
  ignored = motions_dir / "fallAndGetUp2_subject2.npz"
  selected.write_bytes(b"selected")
  ignored.write_bytes(b"ignored")
  (demo_dir / "source_gate.json").write_text('{"status": "GO"}')
  (demo_dir / "manifest.json").write_text(
    json.dumps(
      {
        "schema_version": "g1-getup-amp-v1",
        "accepted_count": 2,
        "rejected_count": 0,
        "accepted": [
          {
            "path": str(selected),
            "accepted": True,
            "reason": "accepted",
            "output_path": str(selected),
            "metadata": {},
          },
          {
            "path": str(ignored),
            "accepted": True,
            "reason": "accepted",
            "output_path": str(ignored),
            "metadata": {},
          },
        ],
        "rejected": [],
      }
    )
  )

  selected_manifest = write_selected_train_manifest(
    demo_data_dir=str(demo_dir),
    npz_files=[str(selected)],
  )

  manifest = json.loads(selected_manifest.read_text())
  assert selected_manifest == demo_dir / "selected_manifest.json"
  assert manifest["accepted_count"] == 1
  assert manifest["accepted"][0]["output_path"] == str(selected)
  assert ignored.name not in json.dumps(manifest)
