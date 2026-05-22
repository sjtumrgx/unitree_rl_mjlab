from __future__ import annotations

import pytest
import torch

from scripts import evaluate_getup_amp_style as style


def test_nearest_neighbor_style_distance_uses_expert_normalization() -> None:
  expert = torch.tensor(
    [
      [1.0, 10.0],
      [3.0, 14.0],
      [5.0, 18.0],
    ]
  )
  policy = torch.tensor(
    [
      [3.0, 14.0],
      [4.0, 16.0],
    ]
  )

  result = style.nearest_neighbor_style_distance(policy, expert)

  assert result["policy_frame_count"] == 2
  assert result["expert_frame_count"] == 3
  assert result["nearest_distance_min"] == pytest.approx(0.0)
  assert result["nearest_distance_mean"] > 0.0
  assert result["nearest_distance_p90"] >= result["nearest_distance_mean"]


def test_compare_style_reports_positive_improvement_when_candidate_is_closer() -> None:
  report = style.build_comparison_report(
    candidate={
      "checkpoint_file": "candidate.pt",
      "style": {"nearest_distance_mean": 0.5},
      "rollout_summary": {"success": {"single_episode_success_rate": 1.0}},
    },
    baseline={
      "checkpoint_file": "baseline.pt",
      "style": {"nearest_distance_mean": 0.8},
      "rollout_summary": {"success": {"single_episode_success_rate": 1.0}},
    },
    success_threshold=0.95,
  )

  assert report["style_improvement_vs_baseline"] == pytest.approx(0.3)
  assert report["style_distance_improved"] is True
  assert report["candidate_success_rate"] == pytest.approx(1.0)
  assert report["style_gate_pass"] is True
