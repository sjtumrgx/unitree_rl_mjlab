"""Shared curriculum constants/config for Unitree G1 anti-fall training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CURRICULUM_TASK_ID = "Unitree-G1-AntiFall-Curriculum"
ANTI_FALL_STAGE_TASK_IDS = (
  "Unitree-G1-AntiFall-Stage0",
  "Unitree-G1-AntiFall-Stage1",
  "Unitree-G1-AntiFall-Stage2",
  "Unitree-G1-AntiFall-Stage3",
  "Unitree-G1-AntiFall-Stage4a",
  "Unitree-G1-AntiFall-Stage4b",
)


@dataclass
class AntiFallCurriculumCfg:
  """Configuration for the curriculum runner."""

  stage_task_ids: tuple[str, ...] = ANTI_FALL_STAGE_TASK_IDS
  per_stage_max_iterations: int = 10000
  rolling_window_iterations: int = 20
  min_iterations_before_promotion: int = 100
  stage0_controllable_locomotion_threshold: float = 0.95
  recovery_rate_threshold: float = 0.80
  recovery_latency_threshold_s: float = 1.50
  min_disturbances_in_window: float = 10.0
  evaluation_interval: int = 1
  load_mode: Literal["auto", "full", "actor_only"] = "auto"
  benchmark_on_promotion: bool = False
  stop_on_unhealthy_run: bool = True
  force_promotion_after_iterations: int | None = None


def curriculum_stage_name(task_id: str) -> str:
  marker = "AntiFall-"
  if marker not in task_id:
    return task_id.lower().replace(" ", "-")
  return task_id.split(marker, 1)[1].lower()
