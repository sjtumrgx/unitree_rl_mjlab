from __future__ import annotations

from pathlib import Path

from scripts import diagnose_getup_multiterrain as multi


def _summary(rate: float, *, risk: bool = False) -> dict:
  return {
    "type": "summary",
    "status": "ok",
    "success": {
      "single_episode_success_rate": rate,
      "success_count_estimate": int(rate * 100),
    },
    "risk_flags": {
      "target_delta_gt_1rad": risk,
      "supportless_height_spike": False,
    },
  }


def test_aggregate_multiterrain_success_requires_every_terrain() -> None:
  report = multi.aggregate_terrain_summaries(
    checkpoint_file="model.pt",
    terrains={
      "ground": _summary(1.0),
      "platform": _summary(0.96),
      "wall": _summary(0.94, risk=True),
    },
    success_threshold=0.95,
  )

  assert report["schema_version"] == multi.SCHEMA_VERSION
  assert report["checkpoint_file"] == "model.pt"
  assert report["success_threshold"] == 0.95
  assert report["all_terrains_success"] is False
  assert report["weakest_terrain"] == "wall"
  assert report["terrain_success_rates"] == {
    "ground": 1.0,
    "platform": 0.96,
    "wall": 0.94,
  }
  assert report["combined_risk_flags"]["target_delta_gt_1rad"] is True


def test_build_rollout_args_sets_getup_terrain_and_common_runtime_fields() -> None:
  args = multi.build_rollout_args(
    terrain="slope",
    checkpoint_file=Path("ckpt.pt"),
    agent="trained",
    num_envs=16,
    steps=300,
    device="cuda:1",
    stop_on_done=True,
  )

  assert args.task_id == "Unitree-G1-GetUp"
  assert args.getup_terrain == "slope"
  assert args.checkpoint_file == "ckpt.pt"
  assert args.num_envs == 16
  assert args.steps == 300
  assert args.device == "cuda:1"
  assert args.stop_on_done is True
