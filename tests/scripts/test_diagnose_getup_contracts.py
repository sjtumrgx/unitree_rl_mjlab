from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_contracts(tmp_path: Path) -> dict:
  output = tmp_path / "contracts.json"
  completed = subprocess.run(
    [
      sys.executable,
      "scripts/diagnose_getup_contracts.py",
      "--output",
      str(output),
    ],
    check=True,
    cwd=Path(__file__).resolve().parents[2],
    text=True,
    capture_output=True,
  )
  stdout = json.loads(completed.stdout)
  written = json.loads(output.read_text())
  assert stdout == written
  return written


def test_diagnose_getup_contracts_reports_action_envelope_and_env_delta_cap(tmp_path: Path) -> None:
  diagnostic = _run_contracts(tmp_path)
  envelope = diagnostic["action_envelope"]

  assert envelope["clip_actions"] == 5.0
  assert envelope["action_cfg_scale"] == 1.0
  assert envelope["initial_action_rescale"] == 1.0
  assert envelope["max_policy_delta_rad"] == 5.0
  assert envelope["action_cfg_max_delta_rad"] == 1.0
  assert envelope["max_env_delta_rad"] == 1.0
  assert envelope["risk_level"] == "mitigated"
  assert envelope["telemetry_required"] is False
  assert envelope["risk_only_not_fix_trigger"] is True


def test_diagnose_getup_contracts_reports_train_play_assist_mismatch_explicitly(tmp_path: Path) -> None:
  diagnostic = _run_contracts(tmp_path)
  assist = diagnostic["assist"]

  assert assist["train_event_present"] is True
  assert assist["play_event_present"] is False
  assert assist["train_play_mismatch_expected"] is True
  assert set(assist["metrics"]) == {"getup_assist_force_n", "getup_action_rescale"}
  assert assist["stable_success_required"] is True
  assert assist["initial_force_n"] == 100
  assert assist["force_decay_n"] == 20.0
  assert assist["action_scale_decay"] == 0.02


def test_diagnose_getup_contracts_reports_reset_and_amp_contracts(tmp_path: Path) -> None:
  diagnostic = _run_contracts(tmp_path)

  assert diagnostic["reset"]["fail_closed_required"] is True
  assert len(diagnostic["reset"]["base_presets"]) == 4
  assert diagnostic["amp"]["env_enabled"] is True
  assert diagnostic["amp"]["terrain"] == "ground"
  assert diagnostic["amp"]["obs_dim_expected"] == 51
  assert diagnostic["amp"]["canonical_joint_count"] == 23
  assert diagnostic["amp"]["amp_obs_group"] == "amp"
