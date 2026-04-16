from __future__ import annotations

import torch

from src.tasks.velocity.rl.antifall_curriculum import AntiFallCurriculumCfg
from src.tasks.velocity.rl.curriculum_metrics import StagePromotionMonitor


def _step_values(metric_names: tuple[str, ...], **values: float) -> torch.Tensor:
  return torch.tensor(
    [[values.get(name, 0.0) for name in metric_names]],
    dtype=torch.float,
  )


def test_stage0_promotion_uses_controllable_locomotion_threshold() -> None:
  cfg = AntiFallCurriculumCfg(
    rolling_window_iterations=2,
    min_iterations_before_promotion=2,
    stage0_controllable_locomotion_threshold=0.95,
  )
  monitor = StagePromotionMonitor("stage0", cfg)
  metric_names = ("controllable_locomotion",)

  monitor.observe_step(metric_names, _step_values(metric_names, controllable_locomotion=0.97))
  monitor.finish_iteration(1)
  assert not monitor.evaluate(1).promote

  monitor.observe_step(metric_names, _step_values(metric_names, controllable_locomotion=0.99))
  monitor.finish_iteration(2)
  decision = monitor.evaluate(2)
  assert decision.promote is True
  assert decision.reason == "threshold_met"
  assert decision.metrics is not None
  assert decision.metrics["controllable_locomotion"] >= 0.95


def test_recovery_stage_promotion_uses_recovery_rate_and_latency() -> None:
  cfg = AntiFallCurriculumCfg(
    rolling_window_iterations=2,
    min_iterations_before_promotion=2,
    recovery_rate_threshold=0.80,
    recovery_latency_threshold_s=1.50,
    min_disturbances_in_window=2.0,
  )
  monitor = StagePromotionMonitor("stage2", cfg)
  metric_names = (
    "disturbance_count",
    "recovery_success_count",
    "recovery_latency",
  )

  for iteration in (1, 2):
    monitor.observe_step(
      metric_names,
      _step_values(
        metric_names,
        disturbance_count=2.0,
        recovery_success_count=2.0,
        recovery_latency=1.0,
      ),
    )
    monitor.finish_iteration(iteration)

  decision = monitor.evaluate(2)
  assert decision.promote is True
  assert decision.reason == "threshold_met"
  assert decision.metrics is not None
  assert decision.metrics["recovery_rate"] >= 0.80
  assert decision.metrics["recovery_latency_s"] == 1.0


def test_force_promotion_and_health_stop_are_explicit() -> None:
  cfg = AntiFallCurriculumCfg(
    rolling_window_iterations=1,
    min_iterations_before_promotion=100,
    force_promotion_after_iterations=3,
  )
  monitor = StagePromotionMonitor("stage4b", cfg)
  metric_names = ("disturbance_count",)
  monitor.observe_step(metric_names, _step_values(metric_names, disturbance_count=0.0))
  monitor.finish_iteration(3)
  assert monitor.evaluate(3).reason == "forced_for_test"

  unhealthy = monitor.evaluate(3, unhealthy=True)
  assert unhealthy.stop is True
  assert unhealthy.reason == "failed_health_check"
