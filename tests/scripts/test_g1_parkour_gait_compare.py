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



def test_compare_gait_logs_reports_first_divergence_for_raw_action(tmp_path) -> None:
  py_log = tmp_path / "python.jsonl"
  cpp_log = tmp_path / "cpp.jsonl"
  out = tmp_path / "summary.json"
  base = {
    "command": [0.25, 0.0, 0.0],
    "base_ang_vel": [0.0, 0.0, 0.0],
    "projected_gravity": [0.0, 0.0, -1.0],
    "raw_action_policy_order": [0.0, 0.0],
    "applied_action_deploy_order": [0.0, 0.0],
    "target_q_deploy_order": [0.1, -0.1],
    "joint_pos_deploy_order": [0.1, -0.1],
    "joint_vel_deploy_order": [0.0, 0.0],
  }
  py_rows = [
    {**base, "step": 0},
    {**base, "step": 1, "raw_action_policy_order": [0.5, -0.5]},
  ]
  cpp_rows = [
    {**base, "step": 0},
    {**base, "step": 1, "raw_action_policy_order": [0.0, 0.0]},
  ]
  _write_jsonl(py_log, py_rows)
  _write_jsonl(cpp_log, cpp_rows)

  subprocess.run(
    [
      sys.executable,
      str(SCRIPT),
      "--python-jsonl",
      str(py_log),
      "--cpp-jsonl",
      str(cpp_log),
      "--output-json",
      str(out),
      "--field-threshold",
      "raw_action_policy_order=0.1",
    ],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
  )

  payload = json.loads(out.read_text(encoding="utf-8"))
  assert payload["first_divergence"]["field"] == "raw_action_policy_order"
  assert payload["first_divergence"]["index"] == 1
  assert payload["first_divergence"]["step"] == 1
  assert payload["first_divergence"]["abs"] > payload["first_divergence"]["threshold"]
  assert payload["fields"]["raw_action_policy_order"]["first_divergence_index"] == 1
  assert payload["fields"]["raw_action_policy_order"]["verdict"] == "diverged"


def test_compare_gait_logs_reports_no_first_divergence_when_diffs_stay_within_threshold(tmp_path) -> None:
  py_log = tmp_path / "python.jsonl"
  cpp_log = tmp_path / "cpp.jsonl"
  out = tmp_path / "summary.json"
  base = {
    "command": [0.25, 0.0, 0.0],
    "base_ang_vel": [0.0, 0.0, 0.0],
    "projected_gravity": [0.0, 0.0, -1.0],
    "raw_action_policy_order": [0.01, -0.01],
    "applied_action_deploy_order": [0.01, -0.01],
    "target_q_deploy_order": [0.1, -0.1],
    "joint_pos_deploy_order": [0.1, -0.1],
    "joint_vel_deploy_order": [0.0, 0.0],
  }
  _write_jsonl(py_log, [{**base, "step": 0}, {**base, "step": 1}])
  _write_jsonl(cpp_log, [{**base, "step": 0}, {**base, "step": 1}])

  subprocess.run(
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
  assert payload["first_divergence"] is None
  assert payload["early_first_divergence"] is None
  assert payload["fields"]["raw_action_policy_order"]["verdict"] == "pass"
  assert payload["diagnosis"]["likely_source"] == "no large early mismatch"


def test_compare_gait_logs_summarizes_cpp_history_freshness(tmp_path) -> None:
  py_log = tmp_path / "python.jsonl"
  cpp_log = tmp_path / "cpp.jsonl"
  out = tmp_path / "summary.json"
  base = {
    "command": [0.25, 0.0, 0.0],
    "base_ang_vel": [0.0, 0.0, 0.0],
    "projected_gravity": [0.0, 0.0, -1.0],
    "raw_action_policy_order": [0.0, 0.0],
    "applied_action_deploy_order": [0.0, 0.0],
    "target_q_deploy_order": [0.1, -0.1],
    "joint_pos_deploy_order": [0.1, -0.1],
    "joint_vel_deploy_order": [0.0, 0.0],
  }
  _write_jsonl(py_log, [{**base, "step": 0}, {**base, "step": 1}])
  _write_jsonl(
    cpp_log,
    [
      {
        **base,
        "step": 0,
        "lowstate_tick": 100,
        "sim_time": 0.1,
        "policy_wall_time": 0.0,
        "history_freshness": {
          "lowstate_ticks": [100],
          "tick_deltas": [],
          "expected_tick_delta": 20,
          "repeated_frame_count": 0,
          "skipped_tick_count": 0,
          "last_action_age_steps": 0,
          "reset_epoch": 1,
        },
      },
      {
        **base,
        "step": 1,
        "lowstate_tick": 120,
        "sim_time": 0.12,
        "policy_wall_time": 0.02,
        "history_freshness": {
          "lowstate_ticks": [100, 120],
          "tick_deltas": [20],
          "expected_tick_delta": 20,
          "repeated_frame_count": 0,
          "skipped_tick_count": 0,
          "last_action_age_steps": 0,
          "reset_epoch": 1,
        },
      },
    ],
  )

  subprocess.run(
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
  freshness = payload["cpp_freshness"]
  assert freshness["verdict"] == "pass"
  assert freshness["samples_with_history"] == 2
  assert freshness["expected_tick_delta"] == 20
  assert freshness["lowstate_tick_delta"]["median"] == 20
  assert freshness["sim_time_delta"]["median"] == 0.01999999999999999
  assert freshness["reset_epochs"] == [1]


def test_compare_gait_logs_flags_repeated_or_skipped_cpp_history_frames(tmp_path) -> None:
  py_log = tmp_path / "python.jsonl"
  cpp_log = tmp_path / "cpp.jsonl"
  out = tmp_path / "summary.json"
  base = {
    "command": [0.25, 0.0, 0.0],
    "base_ang_vel": [0.0, 0.0, 0.0],
    "projected_gravity": [0.0, 0.0, -1.0],
    "raw_action_policy_order": [0.0, 0.0],
    "applied_action_deploy_order": [0.0, 0.0],
    "target_q_deploy_order": [0.1, -0.1],
    "joint_pos_deploy_order": [0.1, -0.1],
    "joint_vel_deploy_order": [0.0, 0.0],
  }
  _write_jsonl(py_log, [{**base, "step": 0}, {**base, "step": 1}])
  _write_jsonl(
    cpp_log,
    [
      {
        **base,
        "step": 0,
        "lowstate_tick": 100,
        "history_freshness": {
          "expected_tick_delta": 20,
          "repeated_frame_count": 0,
          "skipped_tick_count": 0,
          "last_action_age_steps": 0,
          "reset_epoch": 1,
        },
      },
      {
        **base,
        "step": 1,
        "lowstate_tick": 100,
        "history_freshness": {
          "expected_tick_delta": 20,
          "repeated_frame_count": 1,
          "skipped_tick_count": 1,
          "last_action_age_steps": 2,
          "reset_epoch": 1,
        },
      },
    ],
  )

  subprocess.run(
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
  freshness = payload["cpp_freshness"]
  assert freshness["verdict"] == "fail"
  assert freshness["repeated_lowstate_tick_count"] == 1
  assert freshness["max_history_repeated_frame_count"] == 1
  assert freshness["max_history_skipped_tick_count"] == 1
  assert freshness["max_last_action_age_steps"] == 2


def test_compare_gait_logs_flags_cpp_lowstate_tick_jitter(tmp_path) -> None:
  py_log = tmp_path / "python.jsonl"
  cpp_log = tmp_path / "cpp.jsonl"
  out = tmp_path / "summary.json"
  base = {
    "command": [0.25, 0.0, 0.0],
    "base_ang_vel": [0.0, 0.0, 0.0],
    "projected_gravity": [0.0, 0.0, -1.0],
    "raw_action_policy_order": [0.0, 0.0],
    "applied_action_deploy_order": [0.0, 0.0],
    "target_q_deploy_order": [0.1, -0.1],
    "joint_pos_deploy_order": [0.1, -0.1],
    "joint_vel_deploy_order": [0.0, 0.0],
  }
  _write_jsonl(py_log, [{**base, "step": 0}, {**base, "step": 1}])
  _write_jsonl(
    cpp_log,
    [
      {
        **base,
        "step": 0,
        "lowstate_tick": 100,
        "history_freshness": {
          "expected_tick_delta": 20,
          "repeated_frame_count": 0,
          "skipped_tick_count": 0,
          "last_action_age_steps": 0,
          "reset_epoch": 1,
        },
      },
      {
        **base,
        "step": 1,
        "lowstate_tick": 115,
        "history_freshness": {
          "expected_tick_delta": 20,
          "repeated_frame_count": 0,
          "skipped_tick_count": 0,
          "last_action_age_steps": 0,
          "reset_epoch": 1,
        },
      },
    ],
  )

  subprocess.run(
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

  freshness = json.loads(out.read_text(encoding="utf-8"))["cpp_freshness"]
  assert freshness["verdict"] == "fail"
  assert freshness["lowstate_tick_jitter_count"] == 1
  assert "lowstate tick delta deviated" in freshness["problems"][0]
