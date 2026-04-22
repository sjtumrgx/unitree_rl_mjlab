import json
import subprocess
import sys
from pathlib import Path

from scripts.topology_getup_harness import default_scenarios_for_task


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


def test_stage0_default_scenarios_include_three_holdout_topology_families() -> None:
  scenarios = default_scenarios_for_task("Unitree-G1-TopologyGetUp-Stage0")
  heldout = [scenario for scenario in scenarios if "heldout" in scenario.tags]
  assert len(heldout) == 3
  names = {scenario.name for scenario in heldout}
  assert names == {
    "stair-height-heldout",
    "edge-geometry-heldout",
    "support-arrangement-heldout",
  }
  assert {scenario.bucket for scenario in heldout} == {
    "stair-height-heldout",
    "edge-geometry-heldout",
    "support-arrangement-heldout",
  }


def test_benchmark_cli_emits_expected_contracts() -> None:
  repo_root = _repo_root()
  scenarios = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "scenarios",
      "Unitree-G1-TopologyGetUp-Benchmark",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(scenarios.stdout)
  assert payload["stage_name"] == "benchmark"
  assert len(payload["scenarios"]) == 3
  assert all("heldout" in scenario["tags"] for scenario in payload["scenarios"])
  assert {scenario["bucket"] for scenario in payload["scenarios"]} == {
    "stair-height-heldout",
    "edge-geometry-heldout",
    "support-arrangement-heldout",
  }

  smoke = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "smoke-command",
      "Unitree-G1-TopologyGetUp-Stage0",
      "--seed",
      "7",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  smoke_payload = json.loads(smoke.stdout)
  assert smoke_payload["command"][0] == sys.executable
  assert smoke_payload["command"][1] == "scripts/train.py"
  assert smoke_payload["command"][2] == "Unitree-G1-TopologyGetUp-Stage0"


def test_distillation_task_reuses_stage0_topology_scenarios() -> None:
  scenarios = default_scenarios_for_task("Unitree-G1-TopologyGetUp-Stage0-Distill")
  assert any(scenario.name == "stair-height-heldout" for scenario in scenarios)
  assert len(scenarios) == 5


def test_robot_checklist_cli_emits_expected_trial_matrix() -> None:
  repo_root = _repo_root()
  result = subprocess.run(
    [sys.executable, "scripts/benchmark_topology_getup.py", "robot-checklist"],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  trial_names = [item["name"] for item in payload["robot_trials"]]
  assert trial_names == [
    "sensor-availability",
    "sgi-capture-dry-run",
    "flat-getup",
    "slope-getup",
    "edge-like-support",
    "degraded-depth",
  ]


def test_compare_summary_cli_reports_margin_pass(tmp_path: Path) -> None:
  repo_root = _repo_root()
  main_summary = tmp_path / "main.json"
  baseline_summary = tmp_path / "baseline.json"
  main_summary.write_text(json.dumps({
    "aggregate_success_rate": 0.72,
    "heldout_by_bucket": {
      "stair-height-heldout": {"success_rate": 0.72},
      "edge-geometry-heldout": {"success_rate": 0.68},
      "support-arrangement-heldout": {"success_rate": 0.71},
    },
  }))
  baseline_summary.write_text(json.dumps({
    "aggregate_success_rate": 0.60,
    "heldout_by_bucket": {
      "stair-height-heldout": {"success_rate": 0.58},
      "edge-geometry-heldout": {"success_rate": 0.57},
      "support-arrangement-heldout": {"success_rate": 0.56},
    },
  }))
  result = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "compare-summary",
      str(main_summary),
      str(baseline_summary),
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  assert payload["aggregate_pass"] is True
  assert payload["bucket_pass"]["stair-height-heldout"] is True
  assert payload["passed"] is True


def test_compare_summary_cli_requires_all_heldout_buckets_to_pass(tmp_path: Path) -> None:
  repo_root = _repo_root()
  main_summary = tmp_path / "main.json"
  baseline_summary = tmp_path / "baseline.json"
  main_summary.write_text(json.dumps({
    "aggregate_success_rate": 0.72,
    "heldout_by_bucket": {
      "stair-height-heldout": {"success_rate": 0.72},
      "edge-geometry-heldout": {"success_rate": 0.59},
    },
  }))
  baseline_summary.write_text(json.dumps({
    "aggregate_success_rate": 0.60,
    "heldout_by_bucket": {
      "stair-height-heldout": {"success_rate": 0.58},
      "edge-geometry-heldout": {"success_rate": 0.57},
    },
  }))
  result = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "compare-summary",
      str(main_summary),
      str(baseline_summary),
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  assert payload["aggregate_pass"] is True
  assert payload["bucket_pass"]["stair-height-heldout"] is True
  assert payload["bucket_pass"]["edge-geometry-heldout"] is False
  assert payload["passed"] is False


def test_compare_summary_cli_rejects_missing_required_heldout_family(tmp_path: Path) -> None:
  repo_root = _repo_root()
  main_summary = tmp_path / "main.json"
  baseline_summary = tmp_path / "baseline.json"
  main_summary.write_text(json.dumps({
    "aggregate_success_rate": 0.72,
    "heldout_by_bucket": {
      "stair-height-heldout": {"success_rate": 0.72},
      "edge-geometry-heldout": {"success_rate": 0.68},
    },
  }))
  baseline_summary.write_text(json.dumps({
    "aggregate_success_rate": 0.60,
    "heldout_by_bucket": {
      "stair-height-heldout": {"success_rate": 0.58},
      "edge-geometry-heldout": {"success_rate": 0.57},
      "support-arrangement-heldout": {"success_rate": 0.56},
    },
  }))
  result = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "compare-summary",
      str(main_summary),
      str(baseline_summary),
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  assert payload["missing_buckets"] == ["support-arrangement-heldout"]
  assert payload["passed"] is False


def test_suite_plan_cli_emits_teacher_main_naive_and_distill_lanes() -> None:
  repo_root = _repo_root()
  result = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "suite-plan",
      "--teacher-checkpoint",
      "/tmp/teacher.pt",
      "--iterations",
      "12",
      "--num-envs",
      "32",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  lanes = payload["lanes"]
  assert [lane["lane"] for lane in lanes] == ["teacher", "main", "naive_depth", "distill"]
  assert lanes[0]["command"][1] == "scripts/train_topology_getup_teacher.py"
  assert lanes[1]["command"][1] == "scripts/train_topology_getup_main.py"
  assert lanes[2]["command"][1] == "scripts/train_topology_getup_naive.py"
  assert lanes[3]["command"][1] == "scripts/train_topology_getup_distill.py"
  assert "/tmp/teacher.pt" in lanes[3]["command"]


def test_suite_plan_cli_accepts_teacher_run_dir_for_distill_lane() -> None:
  repo_root = _repo_root()
  result = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "suite-plan",
      "--teacher-run-dir",
      "/tmp/teacher_run",
      "--iterations",
      "12",
      "--num-envs",
      "32",
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  distill_lane = payload["lanes"][3]
  assert distill_lane["lane"] == "distill"
  assert distill_lane["command"][1] == "scripts/train_topology_getup_distill.py"
  assert "--teacher-run-dir" in distill_lane["command"]
  assert "/tmp/teacher_run" in distill_lane["command"]


def test_aggregate_summary_cli_emits_weighted_bucket_summary(tmp_path: Path) -> None:
  repo_root = _repo_root()
  results_json = tmp_path / "results.json"
  results_json.write_text(json.dumps([
    {"bucket": "stair-height-heldout", "success_rate": 0.8, "episodes": 2},
    {"bucket": "stair-height-heldout", "success_rate": 0.4, "episodes": 1},
    {"bucket": "edge-geometry-heldout", "success_rate": 0.5, "episodes": 1},
  ]))
  result = subprocess.run(
    [
      sys.executable,
      "scripts/benchmark_topology_getup.py",
      "aggregate-summary",
      str(results_json),
    ],
    cwd=repo_root,
    check=True,
    capture_output=True,
    text=True,
  )
  payload = json.loads(result.stdout)
  assert round(payload["aggregate_success_rate"], 3) == 0.625
  assert round(payload["heldout_by_bucket"]["stair-height-heldout"]["success_rate"], 3) == 0.667
  assert payload["heldout_by_bucket"]["stair-height-heldout"]["episodes"] == 3
