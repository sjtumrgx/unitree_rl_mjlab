from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "play_topology_getup.py"

ALLOWED_TASKS = (
  "Unitree-G1-TopologyGetUp-Stage0",
  "Unitree-G1-TopologyGetUp-Benchmark",
  "Unitree-G1-TopologyGetUp-Stage0-NaiveDepth",
  "Unitree-G1-TopologyGetUp-Stage0-Distill",
)

REJECTED_TASKS = (
  (
    "Unitree-G1-AntiFall-Stage0",
    "Non-TopologyGetUp task",
  ),
  (
    "Unitree-G1-TopologyGetUp-Unknown",
    "Unsupported TopologyGetUp task",
  ),
)


@dataclass
class CapturingPlayConfig:
  agent: str = "WRONG_AGENT"
  checkpoint_file: str | None = None
  motion_file: str | None = "UNEXPECTED_MOTION"
  num_envs: int | None = 999
  device: str | None = "WRONG_DEVICE"
  keyboard_impulse: bool = False
  video: bool = True
  video_length: int = 999
  video_height: int | None = 999
  video_width: int | None = 999
  camera: int | str | None = "WRONG_CAMERA"
  viewer: str = "WRONG_VIEWER"
  no_terminations: bool = True
  _demo_mode: bool = True


@dataclass
class RecordingPlayCall:
  task_id: str
  cfg: CapturingPlayConfig


def _load_wrapper_module(module_name: str | None = None):
  if not SCRIPT_PATH.exists():
    pytest.skip("scripts/play_topology_getup.py is not present in this lane")

  unique_name = module_name or f"test_play_topology_getup_runtime_{len(sys.modules)}"
  sys.modules.pop(unique_name, None)

  spec = importlib.util.spec_from_file_location(unique_name, SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _install_play_stub(monkeypatch: pytest.MonkeyPatch, calls: list[RecordingPlayCall]) -> None:
  fake_play = types.ModuleType("scripts.play")

  def fake_run_play(task_id: str, cfg: CapturingPlayConfig) -> None:
    calls.append(RecordingPlayCall(task_id=task_id, cfg=cfg))

  fake_play.PlayConfig = CapturingPlayConfig
  fake_play.run_play = fake_run_play
  monkeypatch.setitem(sys.modules, "scripts.play", fake_play)


def _invoke_main(module, monkeypatch: pytest.MonkeyPatch, args: list[str]):
  main = getattr(module, "main")
  parameters = inspect.signature(main).parameters
  if len(parameters) == 0:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), *args])
    return main()
  return main(args)


def _invoke_failure(module, monkeypatch: pytest.MonkeyPatch, args: list[str]):
  try:
    _invoke_main(module, monkeypatch, args)
  except BaseException as exc:  # noqa: BLE001
    return exc
  pytest.fail("Expected play_topology_getup invocation to fail")


def _failure_text(exc: BaseException, capsys: pytest.CaptureFixture[str]) -> str:
  captured = capsys.readouterr()
  return "\n".join(part for part in (str(exc), captured.out, captured.err) if part)


def test_defaults_to_stage0_trained_native_when_only_checkpoint_is_given(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  checkpoint = tmp_path / "policy.pt"
  checkpoint.write_text("checkpoint")
  calls: list[RecordingPlayCall] = []

  _install_play_stub(monkeypatch, calls)
  monkeypatch.setenv("DISPLAY", ":1")

  module = _load_wrapper_module()
  _invoke_main(module, monkeypatch, ["--checkpoint-file", str(checkpoint)])

  assert len(calls) == 1
  forwarded = calls[0]
  assert forwarded.task_id == "Unitree-G1-TopologyGetUp-Stage0"
  assert forwarded.cfg.agent == "trained"
  assert forwarded.cfg.viewer == "native"
  assert forwarded.cfg.no_terminations is True


@pytest.mark.parametrize("task_id", ALLOWED_TASKS)
def test_accepts_only_approved_topology_getup_tasks(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path, task_id: str
) -> None:
  checkpoint = tmp_path / f"{task_id}.pt"
  checkpoint.write_text("checkpoint")
  calls: list[RecordingPlayCall] = []

  _install_play_stub(monkeypatch, calls)
  monkeypatch.setenv("DISPLAY", ":1")

  module = _load_wrapper_module()
  _invoke_main(module, monkeypatch, ["--checkpoint-file", str(checkpoint), "--task", task_id])

  assert len(calls) == 1
  assert calls[0].task_id == task_id


@pytest.mark.parametrize(("task_id", "expected_message"), REJECTED_TASKS)
def test_rejects_non_topology_getup_tasks(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  capsys: pytest.CaptureFixture[str],
  task_id: str,
  expected_message: str,
) -> None:
  checkpoint = tmp_path / "policy.pt"
  checkpoint.write_text("checkpoint")
  calls: list[RecordingPlayCall] = []

  _install_play_stub(monkeypatch, calls)
  monkeypatch.setenv("DISPLAY", ":1")

  module = _load_wrapper_module()
  exc = _invoke_failure(module, monkeypatch, ["--checkpoint-file", str(checkpoint), "--task", task_id])

  assert not calls
  message = _failure_text(exc, capsys)
  assert task_id in message
  assert expected_message in message
