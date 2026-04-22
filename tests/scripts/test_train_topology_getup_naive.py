from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "train_topology_getup_naive.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_train_topology_getup_naive_{len(sys.modules)}"
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


def test_build_command_targets_naive_task() -> None:
  module = _load_module()
  command = module.build_command(["--agent.max-iterations=5"])
  assert command[0] == sys.executable
  assert command[1] == "scripts/train.py"
  assert command[2] == "Unitree-G1-TopologyGetUp-Stage0-NaiveDepth"
  assert "--agent.max-iterations=5" in command


def test_main_returns_subprocess_code(monkeypatch: pytest.MonkeyPatch) -> None:
  module = _load_module()
  captured = {}

  def fake_run(command, check=False):
    captured["command"] = command
    captured["check"] = check
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(module.subprocess, "run", fake_run)
  result = _invoke_main(module, ["--agent.max-iterations=5"])
  assert result == 0
  assert captured["command"][2] == "Unitree-G1-TopologyGetUp-Stage0-NaiveDepth"
