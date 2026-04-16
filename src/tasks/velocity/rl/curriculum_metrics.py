"""Promotion policy helpers for the anti-fall curriculum runner."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .antifall_curriculum import AntiFallCurriculumCfg


@dataclass
class StageIterationMetrics:
  iteration: int = 0
  step_count: int = 0
  controllable_locomotion_sum: float = 0.0
  disturbance_count: float = 0.0
  recovery_success_count: float = 0.0
  recovery_latency_sum: float = 0.0
  recovery_latency_count: int = 0

  @property
  def controllable_locomotion(self) -> float:
    if self.step_count == 0:
      return 0.0
    return self.controllable_locomotion_sum / self.step_count

  @property
  def recovery_rate(self) -> float:
    if self.disturbance_count <= 0.0:
      return 0.0
    return self.recovery_success_count / self.disturbance_count

  @property
  def recovery_latency_s(self) -> float | None:
    if self.recovery_latency_count <= 0:
      return None
    return self.recovery_latency_sum / self.recovery_latency_count

  def as_dict(self) -> dict[str, float | int | None]:
    payload = asdict(self)
    payload["controllable_locomotion"] = self.controllable_locomotion
    payload["recovery_rate"] = self.recovery_rate
    payload["recovery_latency_s"] = self.recovery_latency_s
    return payload


@dataclass
class PromotionDecision:
  promote: bool = False
  stop: bool = False
  reason: str | None = None
  metrics: dict[str, float | int | None] | None = None


def metrics_from_totals(totals: dict[str, float]) -> dict[str, float | int | None]:
  step_count = int(totals.get("step_count", 0.0))
  disturbance_count = float(totals.get("disturbance_count", 0.0))
  recovery_latency_count = int(totals.get("recovery_latency_count", 0.0))
  controllable_sum = float(totals.get("controllable_locomotion_sum", 0.0))
  recovery_success_count = float(totals.get("recovery_success_count", 0.0))
  recovery_latency_sum = float(totals.get("recovery_latency_sum", 0.0))
  return {
    "iteration": int(totals.get("iteration", 0.0)),
    "step_count": step_count,
    "controllable_locomotion": (controllable_sum / step_count) if step_count else 0.0,
    "disturbance_count": disturbance_count,
    "recovery_success_count": recovery_success_count,
    "recovery_rate": (recovery_success_count / disturbance_count) if disturbance_count else 0.0,
    "recovery_latency_s": (recovery_latency_sum / recovery_latency_count) if recovery_latency_count else None,
  }


def evaluate_promotion(
  stage_name: str,
  cfg: AntiFallCurriculumCfg,
  iteration: int,
  metrics: dict[str, float | int | None],
  *,
  unhealthy: bool = False,
) -> PromotionDecision:
  if unhealthy and cfg.stop_on_unhealthy_run:
    return PromotionDecision(stop=True, reason="failed_health_check", metrics=metrics)
  if cfg.force_promotion_after_iterations is not None and iteration >= cfg.force_promotion_after_iterations:
    return PromotionDecision(promote=True, reason="forced_for_test", metrics=metrics)
  if iteration < cfg.min_iterations_before_promotion:
    return PromotionDecision(metrics=metrics)
  if cfg.evaluation_interval > 1 and iteration % cfg.evaluation_interval != 0:
    return PromotionDecision(metrics=metrics)

  if stage_name == "stage0":
    if float(metrics["controllable_locomotion"]) >= cfg.stage0_controllable_locomotion_threshold:
      return PromotionDecision(promote=True, reason="threshold_met", metrics=metrics)
    return PromotionDecision(metrics=metrics)

  latency_s = metrics["recovery_latency_s"]
  if float(metrics["disturbance_count"]) < cfg.min_disturbances_in_window:
    return PromotionDecision(metrics=metrics)
  if latency_s is None:
    return PromotionDecision(metrics=metrics)
  if (
    float(metrics["recovery_rate"]) >= cfg.recovery_rate_threshold
    and float(latency_s) <= cfg.recovery_latency_threshold_s
  ):
    return PromotionDecision(promote=True, reason="threshold_met", metrics=metrics)
  return PromotionDecision(metrics=metrics)


class StagePromotionMonitor:
  def __init__(self, stage_name: str, cfg: AntiFallCurriculumCfg):
    self.stage_name = stage_name
    self.cfg = cfg
    self._history: deque[StageIterationMetrics] = deque(
      maxlen=max(1, cfg.rolling_window_iterations)
    )
    self._current = StageIterationMetrics()

  def observe_step(self, metric_names: Sequence[str], step_values: torch.Tensor) -> None:
    if step_values.ndim != 2:
      raise ValueError("step_values must be shaped as [num_envs, num_metrics]")
    index = {name: idx for idx, name in enumerate(metric_names)}
    self._current.step_count += 1

    if "controllable_locomotion" in index:
      values = step_values[:, index["controllable_locomotion"]]
      self._current.controllable_locomotion_sum += float(values.mean().item())
    if "disturbance_count" in index:
      values = step_values[:, index["disturbance_count"]]
      self._current.disturbance_count += float(values.sum().item())
    if "recovery_success_count" in index:
      values = step_values[:, index["recovery_success_count"]]
      self._current.recovery_success_count += float(values.sum().item())
    if "recovery_latency" in index:
      values = step_values[:, index["recovery_latency"]]
      nonzero = values[values > 0]
      self._current.recovery_latency_sum += float(nonzero.sum().item())
      self._current.recovery_latency_count += int(nonzero.numel())

  def finish_iteration(self, iteration: int) -> StageIterationMetrics:
    self._current.iteration = iteration
    summary = self._current
    self._history.append(summary)
    self._current = StageIterationMetrics()
    return summary

  def evaluate(self, iteration: int, *, unhealthy: bool = False) -> PromotionDecision:
    metrics = self.aggregate_metrics()
    return evaluate_promotion(
      self.stage_name,
      self.cfg,
      iteration,
      metrics,
      unhealthy=unhealthy,
    )

  def aggregate_totals(self) -> dict[str, float]:
    if not self._history:
      return {
        "iteration": 0.0,
        "step_count": 0.0,
        "controllable_locomotion_sum": 0.0,
        "disturbance_count": 0.0,
        "recovery_success_count": 0.0,
        "recovery_latency_sum": 0.0,
        "recovery_latency_count": 0.0,
      }
    return {
      "iteration": float(self._history[-1].iteration),
      "step_count": float(sum(item.step_count for item in self._history)),
      "controllable_locomotion_sum": float(sum(item.controllable_locomotion_sum for item in self._history)),
      "disturbance_count": float(sum(item.disturbance_count for item in self._history)),
      "recovery_success_count": float(sum(item.recovery_success_count for item in self._history)),
      "recovery_latency_sum": float(sum(item.recovery_latency_sum for item in self._history)),
      "recovery_latency_count": float(sum(item.recovery_latency_count for item in self._history)),
    }

  def aggregate_metrics(self) -> dict[str, float | int | None]:
    return metrics_from_totals(self.aggregate_totals())
