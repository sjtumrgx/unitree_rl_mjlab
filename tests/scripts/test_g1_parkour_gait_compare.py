from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compare_g1_parkour_gait_logs.py"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
  path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_gait_logs_reports_action_mismatch(tmp_path) -> None:
  py_log = tmp_path / "python.jsonl"
  cpp_log = tmp_path / "cpp.jsonl"
  out = tmp_path / "summary.json"
  base = {
    "command": [0.25, 0.0, 0.0],
    "base_ang_vel": [0.0, 0.0, 0.0],
    "projected_gravity": [0.0, 0.0, -1.0],
    "raw_action_deploy_order": [0.0, 0.0],
    "applied_action_deploy_order": [0.0, 0.0],
    "target_q_deploy_order": [0.1, -0.1],
    "joint_pos_deploy_order": [0.1, -0.1],
    "joint_vel_deploy_order": [0.0, 0.0],
  }
  _write_jsonl(py_log, [{**base, "raw_action_policy_order": [0.5, -0.5]}])
  _write_jsonl(cpp_log, [{**base, "raw_action_policy_order": [-0.5, 0.5]}])

  result = subprocess.run(
    [
      sys.executable,
      str(SCRIPT),
      "--python-jsonl",
      str(py_log),
      "--cpp-jsonl",
      str(cpp_log),
      "--output-json",
      str(out),
    ],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
  )

  payload = json.loads(out.read_text(encoding="utf-8"))
  assert payload["status"] == "ok"
  assert payload["aligned_samples"] == 1
  assert payload["fields"]["raw_action_policy_order"]["mae"] == 1.0
  assert payload["diagnosis"]["likely_source"] == "policy input or depth/proprio history"
  assert "policy input" in result.stdout
