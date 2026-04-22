from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "train_topology_getup_distill.py"


def _load_wrapper_module(module_name: str | None = None):
  if not SCRIPT_PATH.exists():
    pytest.skip("scripts/train_topology_getup_distill.py is not present in this lane")

  unique_name = module_name or f"test_train_topology_getup_distill_runtime_{len(sys.modules)}"
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


def test_build_command_forwards_distill_task_and_teacher_checkpoint(tmp_path: Path) -> None:
  module = _load_wrapper_module()
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.write_text("checkpoint")
  command = module.build_command(
    teacher_checkpoint=checkpoint,
    extra_args=["--agent.max-iterations=5", "--gpu-ids", "[0]"],
  )
  assert command[0] == sys.executable
  assert command[1] == "scripts/train.py"
  assert command[2] == "Unitree-G1-TopologyGetUp-Stage0-Distill"
  assert f"--agent.teacher-load-path={checkpoint}" in command
  assert "--agent.max-iterations=5" in command


def test_run_distill_train_validates_checkpoint_and_returns_subprocess_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  module = _load_wrapper_module()
  checkpoint = tmp_path / "teacher.pt"
  checkpoint.write_text("checkpoint")
  captured: dict[str, object] = {}

  def fake_run(command, check=False):
    captured["command"] = command
    captured["check"] = check
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(module.subprocess, "run", fake_run)
  result = module.run_distill_train(teacher_checkpoint=checkpoint, extra_args=["--agent.max-iterations=5"])
  assert result == 0
  assert captured["check"] is False
  command = captured["command"]
  assert command[2] == "Unitree-G1-TopologyGetUp-Stage0-Distill"


def test_main_fails_when_teacher_checkpoint_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
  module = _load_wrapper_module()
  missing_path = ROOT / "missing-teacher.pt"
  with pytest.raises(SystemExit) as exc_info:
    _invoke_main(module, ["--teacher-checkpoint", str(missing_path)])
  assert exc_info.value.code == 2
  assert "Teacher checkpoint file not found" in capsys.readouterr().err


def test_resolve_teacher_checkpoint_prefers_manifest_in_run_dir(tmp_path: Path) -> None:
  module = _load_wrapper_module()
  run_dir = tmp_path / "teacher_run"
  run_dir.mkdir()
  checkpoint = run_dir / "model_42.pt"
  checkpoint.write_text("checkpoint")
  (run_dir / "topology_getup_artifacts.json").write_text(
    json.dumps(
      {
        "schema_version": "topology_getup_artifacts_v1",
        "lane": "teacher",
        "checkpoint": "model_42.pt",
      }
    )
  )

  resolved = module.resolve_teacher_checkpoint(teacher_checkpoint=None, teacher_run_dir=run_dir)
  assert resolved == checkpoint


def test_run_distill_train_accepts_teacher_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  module = _load_wrapper_module()
  run_dir = tmp_path / "teacher_run"
  run_dir.mkdir()
  checkpoint = run_dir / "model_7.pt"
  checkpoint.write_text("checkpoint")
  (run_dir / "topology_getup_artifacts.json").write_text(
    json.dumps(
      {
        "schema_version": "topology_getup_artifacts_v1",
        "lane": "teacher",
        "checkpoint": "model_7.pt",
      }
    )
  )
  captured: dict[str, object] = {}

  def fake_run(command, check=False):
    captured["command"] = command
    captured["check"] = check
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(module.subprocess, "run", fake_run)
  result = module.run_distill_train(
    teacher_checkpoint=None,
    teacher_run_dir=run_dir,
    extra_args=["--agent.max-iterations=5"],
  )
  assert result == 0
  command = captured["command"]
  assert f"--agent.teacher-load-path={checkpoint}" in command
