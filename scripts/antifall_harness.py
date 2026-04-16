from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ControllabilityThresholds:
  linear_velocity_error_xy: float = 0.35
  yaw_rate_error: float = 0.35
  upright_error_xy: float = 0.30
  sustain_steps: int = 10


@dataclass(frozen=True)
class BenchmarkScenario:
  name: str
  bucket: str
  description: str = ""
  level: int | None = None
  difficulty: str | None = None
  episodes: int = 1
  seed: int | None = None
  episode_length_s: float | None = None
  disturbance_onset_s: float | None = None
  env_overrides: dict[str, Any] = field(default_factory=dict)
  tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeAccumulator:
  episode_index: int
  episode_reward: float = 0.0
  steps: int = 0
  sustain_count: int = 0
  recovered_at_s: float | None = None
  max_linear_velocity_error_xy: float = 0.0
  max_yaw_rate_error: float = 0.0
  max_upright_error_xy: float = 0.0
  action_delta_norms: tuple[float, ...] = ()

  def with_step(
    self,
    *,
    reward: float,
    linear_velocity_error_xy: float,
    yaw_rate_error: float,
    upright_error_xy: float,
    sustain_count: int,
    recovered_at_s: float | None,
    action_delta_norm: float | None,
  ) -> "EpisodeAccumulator":
    deltas = self.action_delta_norms
    if action_delta_norm is not None:
      deltas = (*deltas, action_delta_norm)
    return EpisodeAccumulator(
      episode_index=self.episode_index,
      episode_reward=self.episode_reward + reward,
      steps=self.steps + 1,
      sustain_count=sustain_count,
      recovered_at_s=recovered_at_s,
      max_linear_velocity_error_xy=max(
        self.max_linear_velocity_error_xy, linear_velocity_error_xy
      ),
      max_yaw_rate_error=max(self.max_yaw_rate_error, yaw_rate_error),
      max_upright_error_xy=max(self.max_upright_error_xy, upright_error_xy),
      action_delta_norms=deltas,
    )


DEFAULT_FAILURE_TERMS = ("fell_over", "nan_detected")
DEFAULT_RECOVERABLE_SUCCESS_RATE = 0.5
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


def _default_stage_scenarios() -> dict[str, list[BenchmarkScenario]]:
  mild_walk = {
    "commands": {
      "twist": {
        "rel_standing_envs": 0.0,
        "ranges": {
          "lin_vel_x": (0.25, 0.6),
          "lin_vel_y": (-0.15, 0.15),
          "ang_vel_z": (-0.25, 0.25),
        },
      }
    }
  }
  medium_walk = {
    "commands": {
      "twist": {
        "rel_standing_envs": 0.0,
        "ranges": {
          "lin_vel_x": (0.4, 0.9),
          "lin_vel_y": (-0.25, 0.25),
          "ang_vel_z": (-0.4, 0.4),
        },
      }
    }
  }
  standing = {"commands": {"twist": {"rel_standing_envs": 1.0}}}
  return {
    "stage0": [
      BenchmarkScenario(
        name="standing-no-disturbance",
        bucket="standing",
        description="Flat standing baseline with standing commands only.",
        episodes=3,
        seed=100,
        episode_length_s=4.0,
        env_overrides=standing,
        tags=["baseline", "flat", "standing"],
      ),
      BenchmarkScenario(
        name="walking-no-disturbance",
        bucket="walking",
        description="Flat locomotion baseline with mild walking commands.",
        episodes=3,
        seed=200,
        episode_length_s=4.0,
        env_overrides=mild_walk,
        tags=["baseline", "flat", "walking"],
      ),
    ],
    "stage1": [
      BenchmarkScenario(
        name="flat-recovery-l1",
        bucket="flat_push_kick",
        description="Stage 1 frozen flat recovery bucket L1.",
        level=1,
        difficulty="L1",
        episodes=3,
        seed=300,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=mild_walk,
        tags=["flat", "pushkick", "recovery"],
      ),
      BenchmarkScenario(
        name="flat-recovery-l2",
        bucket="flat_push_kick",
        description="Stage 1 frozen flat recovery bucket L2.",
        level=2,
        difficulty="L2",
        episodes=3,
        seed=400,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=medium_walk,
        tags=["flat", "pushkick", "recovery"],
      ),
    ],
    "stage2": [
      BenchmarkScenario(
        name="flat-hard-recovery-l2",
        bucket="flat_hard_recovery",
        description="Stage 2 harder flat recovery bucket L2.",
        level=2,
        difficulty="L2",
        episodes=3,
        seed=500,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=medium_walk,
        tags=["flat", "hard", "recovery"],
      ),
      BenchmarkScenario(
        name="flat-hard-recovery-l3",
        bucket="flat_hard_recovery",
        description="Stage 2 harder flat recovery bucket L3.",
        level=3,
        difficulty="L3",
        episodes=3,
        seed=600,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=medium_walk,
        tags=["flat", "hard", "recovery"],
      ),
    ],
    "stage3": [
      BenchmarkScenario(
        name="rough-mild",
        bucket="rough_terrain",
        description="Stage 3 mild rough-terrain bucket.",
        level=1,
        difficulty="mild",
        episodes=3,
        seed=700,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=mild_walk,
        tags=["rough", "terrain"],
      ),
      BenchmarkScenario(
        name="rough-medium",
        bucket="rough_terrain",
        description="Stage 3 medium rough-terrain bucket.",
        level=2,
        difficulty="medium",
        episodes=3,
        seed=800,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=medium_walk,
        tags=["rough", "terrain"],
      ),
    ],
    "stage4a": [
      BenchmarkScenario(
        name="slip-l1",
        bucket="slip",
        description="Stage 4a slip benchmark bucket L1.",
        level=1,
        difficulty="L1",
        episodes=3,
        seed=900,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=mild_walk,
        tags=["slip"],
      ),
      BenchmarkScenario(
        name="slip-l2",
        bucket="slip",
        description="Stage 4a slip benchmark bucket L2.",
        level=2,
        difficulty="L2",
        episodes=3,
        seed=1000,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=medium_walk,
        tags=["slip"],
      ),
    ],
    "stage4b": [
      BenchmarkScenario(
        name="trip-l1",
        bucket="trip",
        description="Stage 4b trip benchmark bucket L1.",
        level=1,
        difficulty="L1",
        episodes=3,
        seed=1100,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=mild_walk,
        tags=["trip"],
      ),
      BenchmarkScenario(
        name="trip-l2",
        bucket="trip",
        description="Stage 4b trip benchmark bucket L2.",
        level=2,
        difficulty="L2",
        episodes=3,
        seed=1200,
        episode_length_s=6.0,
        disturbance_onset_s=1.0,
        env_overrides=medium_walk,
        tags=["trip"],
      ),
    ],
  }


def normalize_stage_name(task_id: str) -> str | None:
  match = re.search(r"AntiFall-(Stage0|Stage1|Stage2|Stage3|Stage4a|Stage4b|Benchmark)", task_id)
  if not match:
    return None
  return match.group(1).lower()


def default_scenarios_for_task(task_id: str) -> list[BenchmarkScenario]:
  stages = _default_stage_scenarios()
  stage_name = normalize_stage_name(task_id)
  if stage_name is None:
    return []
  if stage_name == "benchmark":
    merged: list[BenchmarkScenario] = []
    for key in ("stage0", "stage1", "stage2", "stage3", "stage4a", "stage4b"):
      merged.extend(stages[key])
    return merged
  return list(stages.get(stage_name, ()))


def ensure_parent(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)


def as_jsonable(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if is_dataclass(value):
    return {key: as_jsonable(val) for key, val in asdict(value).items()}
  if isinstance(value, Mapping):
    return {str(key): as_jsonable(val) for key, val in value.items()}
  if isinstance(value, tuple):
    return [as_jsonable(item) for item in value]
  if isinstance(value, list):
    return [as_jsonable(item) for item in value]
  return value


def write_json(path: Path | str, payload: Any) -> None:
  destination = Path(path)
  ensure_parent(destination)
  destination.write_text(json.dumps(as_jsonable(payload), indent=2, sort_keys=True) + "\n")


def load_json(path: Path | str) -> Any:
  return json.loads(Path(path).read_text())


def load_scenarios_from_json(path: Path | str) -> list[BenchmarkScenario]:
  payload = load_json(path)
  raw_items = payload.get("scenarios", payload) if isinstance(payload, dict) else payload
  if not isinstance(raw_items, list):
    raise TypeError(f"Scenario file must contain a list or {{'scenarios': [...]}}: {path}")
  return [BenchmarkScenario(**item) for item in raw_items]


def apply_overrides(target: Any, overrides: Mapping[str, Any]) -> None:
  for key, override_value in overrides.items():
    current_value: Any
    if isinstance(target, Mapping):
      current_value = target[key]
      if isinstance(current_value, Mapping) and isinstance(override_value, Mapping):
        apply_overrides(current_value, override_value)
      else:
        target[key] = override_value  # type: ignore[index]
      continue

    current_value = getattr(target, key)
    if isinstance(current_value, Mapping) and isinstance(override_value, Mapping):
      apply_overrides(current_value, override_value)
    elif not isinstance(override_value, Mapping):
      setattr(target, key, override_value)
    else:
      apply_overrides(current_value, override_value)


def scalarize(value: Any) -> Any:
  if hasattr(value, "item"):
    try:
      return value.item()
    except Exception:
      pass
  if isinstance(value, dict):
    return {key: scalarize(val) for key, val in value.items()}
  if isinstance(value, (list, tuple)):
    return [scalarize(item) for item in value]
  return value


def summarize_scalars(values: Sequence[float]) -> dict[str, float] | None:
  clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
  if not clean:
    return None
  ordered = sorted(clean)
  index_95 = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
  return {
    "count": float(len(ordered)),
    "min": ordered[0],
    "max": ordered[-1],
    "mean": statistics.fmean(ordered),
    "median": statistics.median(ordered),
    "p95": ordered[index_95],
  }


def extract_termination_counts(log: Mapping[str, Any]) -> dict[str, int]:
  counts: dict[str, int] = {}
  for key, value in log.items():
    if not key.startswith("Episode_Termination/"):
      continue
    term_name = key.split("/", maxsplit=1)[1]
    counts[term_name] = int(scalarize(value) or 0)
  return counts


def infer_failure_terms(
  log: Mapping[str, Any],
  explicit_failure_terms: Sequence[str] | None = None,
) -> tuple[str, ...]:
  if explicit_failure_terms:
    return tuple(explicit_failure_terms)
  termination_counts = extract_termination_counts(log)
  if termination_counts:
    return tuple(name for name in termination_counts if name != "time_out")
  return DEFAULT_FAILURE_TERMS


def compute_control_measure(env: Any, command_name: str) -> dict[str, float]:
  command = env.command_manager.get_command(command_name)
  robot = env.scene["robot"]
  command_xy = command[:, :2]
  command_yaw = command[:, 2]
  actual_lin_xy = robot.data.root_link_lin_vel_b[:, :2]
  actual_yaw = robot.data.root_link_ang_vel_b[:, 2]
  projected_gravity = robot.data.projected_gravity_b[:, :2]
  return {
    "linear_velocity_error_xy": float((command_xy - actual_lin_xy).norm(dim=1)[0].item()),
    "yaw_rate_error": float((command_yaw - actual_yaw).abs()[0].item()),
    "upright_error_xy": float(projected_gravity.square().sum(dim=1).sqrt()[0].item()),
  }


def update_recovery_state(
  *,
  accumulator: EpisodeAccumulator,
  current_time_s: float,
  disturbance_onset_s: float | None,
  thresholds: ControllabilityThresholds,
  control_measure: Mapping[str, float],
) -> tuple[int, float | None]:
  if disturbance_onset_s is None or current_time_s < disturbance_onset_s:
    return accumulator.sustain_count, accumulator.recovered_at_s

  controllable = (
    control_measure["linear_velocity_error_xy"] <= thresholds.linear_velocity_error_xy
    and control_measure["yaw_rate_error"] <= thresholds.yaw_rate_error
    and control_measure["upright_error_xy"] <= thresholds.upright_error_xy
  )
  sustain_count = accumulator.sustain_count + 1 if controllable else 0
  recovered_at_s = accumulator.recovered_at_s
  if (
    recovered_at_s is None
    and sustain_count >= thresholds.sustain_steps
  ):
    recovered_at_s = max(0.0, current_time_s - disturbance_onset_s)
  return sustain_count, recovered_at_s


def summarize_episode_results(
  scenario: BenchmarkScenario,
  episode_results: Sequence[Mapping[str, Any]],
  recoverable_success_rate: float,
) -> dict[str, Any]:
  survived = [1.0 if item["survived"] else 0.0 for item in episode_results]
  no_fall = [1.0 if item["no_fall_success"] else 0.0 for item in episode_results]
  recovered = [
    1.0 if item["recovered_to_controllable_locomotion"] else 0.0
    for item in episode_results
  ]
  recovery_latencies = [
    float(item["recovery_latency_s"])
    for item in episode_results
    if item["recovery_latency_s"] is not None
  ]
  action_spikes = [
    float(item["action_spike_summary"]["max"])
    for item in episode_results
    if item.get("action_spike_summary") and item["action_spike_summary"].get("max") is not None
  ]
  summary = {
    "scenario": scenario.name,
    "bucket": scenario.bucket,
    "level": scenario.level,
    "difficulty": scenario.difficulty,
    "episodes": len(episode_results),
    "survived": statistics.fmean(survived) if survived else 0.0,
    "no_fall_success_rate": statistics.fmean(no_fall) if no_fall else 0.0,
    "recovered_to_controllable_locomotion": statistics.fmean(recovered) if recovered else 0.0,
    "recovery_latency_s": summarize_scalars(recovery_latencies),
    "action_spike_summary": summarize_scalars(action_spikes),
    "max_recoverable_disturbance_level": scenario.level
    if scenario.level is not None
    and (statistics.fmean(recovered) if recovered else 0.0) >= recoverable_success_rate
    else None,
  }
  return summary


def summarize_bucket_results(
  scenario_summaries: Sequence[Mapping[str, Any]],
  recoverable_success_rate: float,
) -> list[dict[str, Any]]:
  buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
  for scenario_summary in scenario_summaries:
    buckets[str(scenario_summary["bucket"])].append(scenario_summary)

  reports: list[dict[str, Any]] = []
  for bucket_name, items in sorted(buckets.items()):
    survived = [float(item["survived"]) for item in items]
    no_fall = [float(item["no_fall_success_rate"]) for item in items]
    recovered = [float(item["recovered_to_controllable_locomotion"]) for item in items]
    latencies: list[float] = []
    candidate_levels: list[int] = []
    for item in items:
      if item.get("recovery_latency_s"):
        latency_summary = item["recovery_latency_s"]
        if latency_summary.get("median") is not None:
          latencies.append(float(latency_summary["median"]))
      if (
        item.get("level") is not None
        and float(item["recovered_to_controllable_locomotion"]) >= recoverable_success_rate
      ):
        candidate_levels.append(int(item["level"]))
    reports.append(
      {
        "bucket": bucket_name,
        "scenario_count": len(items),
        "survived": statistics.fmean(survived) if survived else 0.0,
        "no_fall_success_rate": statistics.fmean(no_fall) if no_fall else 0.0,
        "recovered_to_controllable_locomotion": statistics.fmean(recovered) if recovered else 0.0,
        "recovery_latency_s": summarize_scalars(latencies),
        "max_recoverable_disturbance_level": max(candidate_levels) if candidate_levels else None,
      }
    )
  return reports


def compute_report_max_recoverable_level(
  scenario_summaries: Sequence[Mapping[str, Any]],
  recoverable_success_rate: float,
) -> int | None:
  candidate_levels = [
    int(item["level"])
    for item in scenario_summaries
    if item.get("level") is not None
    and float(item["recovered_to_controllable_locomotion"]) >= recoverable_success_rate
  ]
  return max(candidate_levels) if candidate_levels else None


def build_smoke_command(
  *,
  task_id: str,
  seed: int,
  iterations: int,
  num_envs: int,
  save_interval: int,
  extra_args: Sequence[str] = (),
) -> list[str]:
  return [
    sys.executable,
    "scripts/train.py",
    task_id,
    f"--agent.seed={seed}",
    f"--agent.max-iterations={iterations}",
    f"--agent.save-interval={save_interval}",
    f"--env.scene.num-envs={num_envs}",
    *extra_args,
  ]


def parse_training_health(log_text: str) -> dict[str, Any]:
  lower = log_text.lower()
  nan_matches = sum(bool(pattern.search(lower)) for pattern in _NAN_PATTERNS)
  failure_matches = [pattern.pattern for pattern in _FAILURE_PATTERNS if pattern.search(lower)]
  return {
    "nan_or_corruption_hits": nan_matches,
    "fatal_error_patterns": failure_matches,
    "reward_collapse_flag": nan_matches > 0 or bool(failure_matches),
  }


def extract_log_dir_from_output(output: str) -> Path | None:
  match = _LOG_DIR_PATTERN.search(output)
  if not match:
    return None
  return Path(match.group("path").strip())


def find_latest_run_dir(log_root: Path, started_at: float) -> Path | None:
  if not log_root.exists():
    return None
  candidates = [path for path in log_root.iterdir() if path.is_dir() and path.stat().st_mtime >= started_at - 1.0]
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
  checkpoints = sorted(run_dir.glob("model_*.pt"))
  return checkpoints[-1] if checkpoints else None


def run_subprocess(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
    list(command),
    cwd=str(cwd),
    text=True,
    capture_output=True,
    check=False,
  )
