from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class BenchmarkScenario:
  name: str
  bucket: str
  description: str = ""
  episodes: int = 1
  seed: int | None = None
  episode_length_s: float | None = None
  env_overrides: dict[str, Any] = field(default_factory=dict)
  tags: list[str] = field(default_factory=list)


_STAGE_TASK_IDS = {
  "Unitree-G1-TopologyGetUp-Stage0": "stage0",
  "Unitree-G1-TopologyGetUp-Benchmark": "benchmark",
  "Unitree-G1-TopologyGetUp-Stage0-NaiveDepth": "stage0_naive_depth",
  "Unitree-G1-TopologyGetUp-Stage0-Distill": "stage0_distill",
}

_REQUIRED_HELDOUT_BUCKETS = (
  "stair-height-heldout",
  "edge-geometry-heldout",
  "support-arrangement-heldout",
)


def normalize_stage_name(task_id: str) -> str | None:
  return _STAGE_TASK_IDS.get(task_id)


def default_scenarios_for_task(task_id: str) -> list[BenchmarkScenario]:
  stage_name = normalize_stage_name(task_id)
  if stage_name is None:
    raise ValueError(f"Unsupported topology get-up task id: {task_id}")

  base = [
    BenchmarkScenario(
      name="flat-supine-seen",
      bucket="seen_topology",
      description="Seen flat-support get-up baseline.",
      episodes=2,
      seed=101,
      episode_length_s=8.0,
      tags=["flat", "seen", "supine"],
    ),
    BenchmarkScenario(
      name="slope-prone-seen",
      bucket="seen_topology",
      description="Seen slope-support get-up baseline.",
      episodes=2,
      seed=202,
      episode_length_s=8.0,
      tags=["slope", "seen", "prone"],
    ),
    BenchmarkScenario(
      name="stair-height-heldout",
      bucket="stair-height-heldout",
      description="Held-out stair-height zero-shot evaluation.",
      episodes=2,
      seed=303,
      episode_length_s=8.0,
      tags=["stairs", "heldout", "zero-shot"],
    ),
    BenchmarkScenario(
      name="edge-geometry-heldout",
      bucket="edge-geometry-heldout",
      description="Held-out edge geometry zero-shot evaluation.",
      episodes=2,
      seed=404,
      episode_length_s=8.0,
      tags=["edge", "heldout", "zero-shot"],
    ),
    BenchmarkScenario(
      name="support-arrangement-heldout",
      bucket="support-arrangement-heldout",
      description="Held-out support arrangement around limbs.",
      episodes=2,
      seed=505,
      episode_length_s=8.0,
      tags=["support-arrangement", "heldout", "zero-shot"],
    ),
  ]
  if stage_name == "benchmark":
    return [scenario for scenario in base if "heldout" in scenario.tags]
  if stage_name == "stage0_distill":
    return base
  return base


_LOG_DIR_PATTERN = re.compile(r"Logging experiment in directory:\s*(?P<path>.+)")
_NAN_PATTERNS = (
  re.compile(r"\bnan\b", re.IGNORECASE),
  re.compile(r"\binf\b", re.IGNORECASE),
  re.compile(r"corrupt", re.IGNORECASE),
)
_FAILURE_PATTERNS = (
  re.compile(r"traceback", re.IGNORECASE),
  re.compile(r"runtimeerror", re.IGNORECASE),
  re.compile(r"cuda error", re.IGNORECASE),
  re.compile(r"segmentation fault", re.IGNORECASE),
)


def default_robot_trial_matrix() -> list[dict[str, object]]:
  return [
    {
      "stage": "preflight",
      "name": "sensor-availability",
      "goal": "Confirm onboard depth, IMU, joint states, and contact channels are live.",
      "safety": "Robot on stand or safety tether; no autonomous recovery yet.",
    },
    {
      "stage": "preflight",
      "name": "sgi-capture-dry-run",
      "goal": "Record SGI v1 support patch tensors and verify expected shape/range.",
      "safety": "Robot fixed or manually supported.",
    },
    {
      "stage": "trial",
      "name": "flat-getup",
      "goal": "Tethered flat-ground get-up from preset fallen poses.",
      "safety": "Safety tether, manual e-stop, spotter present.",
    },
    {
      "stage": "trial",
      "name": "slope-getup",
      "goal": "Tethered slope-surface get-up on seen training slope family.",
      "safety": "Safety tether, reduced slope angle first, spotter present.",
    },
    {
      "stage": "trial",
      "name": "edge-like-support",
      "goal": "Edge-like support get-up using held-out edge geometry fixture.",
      "safety": "Safety tether, soft mats, manual reset between trials.",
    },
    {
      "stage": "stress",
      "name": "degraded-depth",
      "goal": "Evaluate robustness under dropped frames / occlusion / restricted view.",
      "safety": "Run only after flat/slope/edge trials pass baseline safety checks.",
    },
  ]


def build_smoke_command(
  *,
  task_id: str,
  seed: int = 1,
  iterations: int = 1,
  num_envs: int = 1,
  save_interval: int = 1,
  extra_args: Sequence[str] = (),
) -> list[str]:
  command = [
    sys.executable,
    "scripts/train.py",
    task_id,
    f"--seed={seed}",
    f"--max_iterations={iterations}",
    f"--save_interval={save_interval}",
    f"--env.scene.num-envs={num_envs}",
  ]
  command.extend(extra_args)
  return command


def build_teacher_command(
  *,
  iterations: int = 1,
  num_envs: int = 1,
  extra_args: Sequence[str] = (),
) -> list[str]:
  command = [
    sys.executable,
    "scripts/train_topology_getup_teacher.py",
    "--",
    f"--agent.max-iterations={iterations}",
    f"--env.scene.num-envs={num_envs}",
  ]
  command.extend(extra_args)
  return command


def build_distill_command(
  *,
  teacher_checkpoint: str | None = None,
  teacher_run_dir: str | None = None,
  iterations: int = 1,
  num_envs: int = 1,
  extra_args: Sequence[str] = (),
) -> list[str]:
  if teacher_checkpoint is None and teacher_run_dir is None:
    raise ValueError("Either teacher_checkpoint or teacher_run_dir must be provided.")
  command = [
    sys.executable,
    "scripts/train_topology_getup_distill.py",
  ]
  if teacher_run_dir is not None:
    command.extend(["--teacher-run-dir", teacher_run_dir])
  else:
    command.extend(["--teacher-checkpoint", str(teacher_checkpoint)])
  command.extend([
    "--",
    f"--agent.max-iterations={iterations}",
    f"--env.scene.num-envs={num_envs}",
  ])
  command.extend(extra_args)
  return command


def default_experiment_suite(
  *,
  teacher_checkpoint: str = "path/to/teacher.pt",
  teacher_run_dir: str | None = None,
  iterations: int = 5000,
  num_envs: int = 4096,
) -> list[dict[str, object]]:
  return [
    {
      "lane": "teacher",
      "task_id": "Unitree-G1-TopologyGetUp-Stage0-Teacher",
      "command": build_teacher_command(iterations=iterations, num_envs=num_envs),
    },
    {
      "lane": "main",
      "task_id": "Unitree-G1-TopologyGetUp-Stage0",
      "command": [
        sys.executable,
        "scripts/train_topology_getup_main.py",
        "--",
        f"--agent.max-iterations={iterations}",
        f"--env.scene.num-envs={num_envs}",
      ],
    },
    {
      "lane": "naive_depth",
      "task_id": "Unitree-G1-TopologyGetUp-Stage0-NaiveDepth",
      "command": [
        sys.executable,
        "scripts/train_topology_getup_naive.py",
        "--",
        f"--agent.max-iterations={iterations}",
        f"--env.scene.num-envs={num_envs}",
      ],
    },
    {
      "lane": "distill",
      "task_id": "Unitree-G1-TopologyGetUp-Stage0-Distill",
      "command": build_distill_command(
        teacher_checkpoint=None if teacher_run_dir is not None else teacher_checkpoint,
        teacher_run_dir=teacher_run_dir,
        iterations=iterations,
        num_envs=num_envs,
      ),
    },
  ]


def parse_training_health(log_text: str) -> dict[str, Any]:
  lower = log_text.lower()
  log_dir = None
  match = _LOG_DIR_PATTERN.search(log_text)
  if match:
    log_dir = match.group("path").strip()
  return {
    "has_nan": any(pattern.search(lower) for pattern in _NAN_PATTERNS),
    "has_failure": any(pattern.search(lower) for pattern in _FAILURE_PATTERNS),
    "log_dir": log_dir,
  }


def aggregate_scenario_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
  if not results:
    raise ValueError("results must not be empty")

  aggregate_success = 0.0
  aggregate_weight = 0
  bucket_totals: dict[str, float] = {}
  bucket_weights: dict[str, int] = {}
  for result in results:
    bucket = str(result["bucket"])
    episodes = int(result.get("episodes", 1))
    success_rate = float(result["success_rate"])
    aggregate_success += success_rate * episodes
    aggregate_weight += episodes
    bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + success_rate * episodes
    bucket_weights[bucket] = bucket_weights.get(bucket, 0) + episodes

  heldout_by_bucket = {
    bucket: {"success_rate": bucket_totals[bucket] / bucket_weights[bucket], "episodes": bucket_weights[bucket]}
    for bucket in bucket_totals
  }
  return {
    "aggregate_success_rate": aggregate_success / aggregate_weight,
    "aggregate_episodes": aggregate_weight,
    "heldout_by_bucket": heldout_by_bucket,
  }


def evaluate_baseline_margin(
  main_summary: dict[str, Any],
  baseline_summary: dict[str, Any],
  *,
  per_bucket_margin: float = 0.10,
  aggregate_margin: float = 0.05,
) -> dict[str, Any]:
  main_buckets = main_summary.get("heldout_by_bucket", {})
  baseline_buckets = baseline_summary.get("heldout_by_bucket", {})
  missing_buckets = sorted(bucket for bucket in _REQUIRED_HELDOUT_BUCKETS if bucket not in main_buckets)
  missing_baseline_buckets = sorted(bucket for bucket in _REQUIRED_HELDOUT_BUCKETS if bucket not in baseline_buckets)
  bucket_deltas: dict[str, float] = {}
  bucket_pass: dict[str, bool] = {}
  for bucket_name in _REQUIRED_HELDOUT_BUCKETS:
    if bucket_name in missing_buckets or bucket_name in missing_baseline_buckets:
      continue
    main_bucket = main_buckets[bucket_name]
    baseline_bucket = baseline_buckets.get(bucket_name)
    if baseline_bucket is None:
      raise ValueError(f"Missing baseline bucket '{bucket_name}' in baseline summary")
    delta = float(main_bucket["success_rate"]) - float(baseline_bucket["success_rate"])
    bucket_deltas[bucket_name] = delta
    bucket_pass[bucket_name] = delta >= per_bucket_margin

  aggregate_delta = float(main_summary["aggregate_success_rate"]) - float(baseline_summary["aggregate_success_rate"])
  return {
    "aggregate_delta": aggregate_delta,
    "aggregate_pass": aggregate_delta >= aggregate_margin,
    "bucket_deltas": bucket_deltas,
    "bucket_pass": bucket_pass,
    "missing_buckets": missing_buckets,
    "missing_baseline_buckets": missing_baseline_buckets,
    "passed": (
      (aggregate_delta >= aggregate_margin)
      and not missing_buckets
      and not missing_baseline_buckets
      and all(bucket_pass.values())
    ),
    "thresholds": {
      "aggregate_margin": aggregate_margin,
      "per_bucket_margin": per_bucket_margin,
    },
  }


def write_json(output: Path, payload: Any) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def json_ready(value: Any) -> Any:
  if is_dataclass(value):
    return {k: json_ready(v) for k, v in asdict(value).items()}
  if isinstance(value, dict):
    return {k: json_ready(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_ready(v) for v in value]
  return value
