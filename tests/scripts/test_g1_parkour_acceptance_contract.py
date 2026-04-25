from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "play_parkour.py"


def _load_module(module_name: str | None = None):
  unique_name = module_name or f"test_g1_parkour_acceptance_contract_{len(sys.modules)}"
  sys.modules.pop(unique_name, None)
  spec = importlib.util.spec_from_file_location(unique_name, SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_timeout_alone_cannot_mark_validate_walk_accepted() -> None:
  module = _load_module()
  source = inspect.getsource(module.run_validate_walk)

  assert "accepted = distance >= args.walk_distance" in source
  assert "accepted = distance >= args.walk_distance or elapsed >= args.max_seconds" not in source
  assert '"status": "ok" if accepted else "failed"' in source
  assert '"duration_target_met": elapsed >= args.max_seconds' in source
  assert "traversal_accepted = distance >= args.walk_distance" in source
  assert "args.depth_contract_only and depth_contract_met" in source


def test_validate_walk_final_diagnostics_refresh_depth_payload() -> None:
  module = _load_module()
  source = inspect.getsource(module.run_validate_walk)

  assert '"depth": depth_provider.diagnostics()' in source
  assert "make_depth_provider(" in source
  assert "debug_dir=args.depth_debug_dir" in source
