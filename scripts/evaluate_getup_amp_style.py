"""Evaluate whether an AMP GetUp policy stays close to demonstration style.

The style metric is intentionally independent from the AMP discriminator score:
it compares rollout AMP observation frames against the expert AMP observation
corpus in the same yaw-invariant 51-D feature space used for training.  Lower
nearest-neighbour distance means the policy visits states closer to the demos.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from scripts import diagnose_getup_rollout as rollout
from src.tasks.velocity.rl.getup_amp_data import AmpExpertDataset

SCHEMA_VERSION = "g1-getup-amp-style-v1"


def _flatten_obs_group(value: Any) -> torch.Tensor:
  if torch.is_tensor(value):
    return value.detach().float().reshape(value.shape[0], -1)
  if isinstance(value, dict):
    parts = [_flatten_obs_group(value[key]) for key in sorted(value)]
    return torch.cat(parts, dim=-1)
  if isinstance(value, (list, tuple)):
    parts = [_flatten_obs_group(item) for item in value]
    return torch.cat(parts, dim=-1)
  raise TypeError(f"Unsupported observation group type: {type(value)!r}")


def _get_obs_group(obs: Any, group_name: str) -> Any:
  if isinstance(obs, dict):
    return obs[group_name]
  return getattr(obs, group_name)


def nearest_neighbor_style_distance(policy_obs: torch.Tensor, expert_obs: torch.Tensor) -> dict[str, float | int]:
  if policy_obs.ndim != 2 or expert_obs.ndim != 2:
    raise ValueError("policy_obs and expert_obs must be rank-2 tensors")
  if policy_obs.shape[1] != expert_obs.shape[1]:
    raise ValueError(
      f"AMP feature dimension mismatch: policy={policy_obs.shape[1]} expert={expert_obs.shape[1]}"
    )
  expert = expert_obs.detach().float().cpu()
  policy = policy_obs.detach().float().cpu()
  mean = expert.mean(dim=0, keepdim=True)
  std = expert.std(dim=0, keepdim=True).clamp_min(1.0e-6)
  expert_n = (expert - mean) / std
  policy_n = (policy - mean) / std
  # cdist is fine for bounded diagnostic rollouts.  The default expert set is
  # small after get-up segment extraction; callers can cap rollout steps/envs.
  distances = torch.cdist(policy_n, expert_n, p=2.0) / (expert_n.shape[1] ** 0.5)
  nearest = distances.min(dim=1).values
  return {
    "policy_frame_count": int(policy.shape[0]),
    "expert_frame_count": int(expert.shape[0]),
    "feature_dim": int(policy.shape[1]),
    "nearest_distance_mean": float(nearest.mean().item()),
    "nearest_distance_median": float(nearest.median().item()),
    "nearest_distance_p90": float(torch.quantile(nearest, 0.9).item()),
    "nearest_distance_min": float(nearest.min().item()),
    "nearest_distance_max": float(nearest.max().item()),
  }


def _load_expert_dataset(args: argparse.Namespace) -> AmpExpertDataset:
  manifest = Path(args.manifest_path).expanduser() if args.manifest_path else Path(args.demo_data_dir).expanduser() / "manifest.json"
  return AmpExpertDataset(
    manifest,
    device=args.device,
    target_dt=float(args.amp_target_dt),
    getup_segments=True,
    feature_layout="yaw_invariant",
  )


def collect_policy_amp_observations(args: argparse.Namespace, checkpoint_file: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  rollout_args = argparse.Namespace(
    task_id="Unitree-G1-GetUp-AMP",
    getup_terrain="ground",
    demo_data_dir=args.demo_data_dir,
    manifest_path=args.manifest_path,
    checkpoint_file=str(checkpoint_file),
    agent="trained",
    num_envs=int(args.num_envs),
    steps=int(args.steps),
    device=args.device,
    train_like=False,
    stop_on_done=bool(args.stop_on_done),
    output=None,
  )
  env_cfg, agent_cfg = rollout._load_getup_configs(rollout_args)
  env_cfg.scene.num_envs = int(args.num_envs)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  frames: list[torch.Tensor] = []
  records: list[dict[str, Any]] = []
  try:
    policy, runner = rollout._make_trained_policy(rollout_args, env, agent_cfg)
    obs = env.get_observations()
    previous_clipped_action = None
    metadata = {
      "schema_version": rollout.SCHEMA_VERSION,
      "type": "metadata",
      "status": "ok",
      "task_id": "Unitree-G1-GetUp-AMP",
      "mode": "play-like",
      "agent": "trained",
      "checkpoint_file": str(checkpoint_file),
      "num_envs": int(args.num_envs),
      "steps_requested": int(args.steps),
      "clip_actions": agent_cfg.clip_actions,
    }
    records.append(metadata)
    for step_index in range(int(args.steps)):
      frames.append(_flatten_obs_group(_get_obs_group(obs, "amp")).detach().cpu())
      with torch.no_grad():
        raw_action = policy(obs)
      clipped = (
        torch.clamp(raw_action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
        if agent_cfg.clip_actions is not None
        else raw_action
      )
      next_obs, rewards, dones, extras = env.step(raw_action)
      amp_stats = rollout._amp_step_stats(runner, obs, next_obs)
      records.append(
        rollout.build_step_record(
          raw_env,
          task_id="Unitree-G1-GetUp-AMP",
          step_index=step_index,
          mode="play-like",
          raw_action=raw_action.detach(),
          clipped_action=clipped.detach(),
          previous_clipped_action=previous_clipped_action,
          rewards=rewards.detach(),
          dones=dones.detach(),
          extras=extras,
          clip_actions=agent_cfg.clip_actions,
          amp_stats=amp_stats,
        )
      )
      previous_clipped_action = clipped.detach().clone()
      obs = next_obs
      if bool(args.stop_on_done) and torch.as_tensor(dones).bool().any().item():
        break
    frames.append(_flatten_obs_group(_get_obs_group(obs, "amp")).detach().cpu())
    records.append(rollout.summarize_records(records))
    return torch.cat(frames, dim=0), records[-1]
  finally:
    env.close()


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_file: str | Path, expert_obs: torch.Tensor) -> dict[str, Any]:
  policy_obs, rollout_summary = collect_policy_amp_observations(args, checkpoint_file)
  return {
    "schema_version": SCHEMA_VERSION,
    "type": "checkpoint_style",
    "status": "ok",
    "checkpoint_file": str(checkpoint_file),
    "style": nearest_neighbor_style_distance(policy_obs, expert_obs),
    "rollout_summary": rollout_summary,
  }


def _success_rate(entry: dict[str, Any]) -> float | None:
  value = entry.get("rollout_summary", {}).get("success", {}).get("single_episode_success_rate")
  return float(value) if value is not None else None


def build_comparison_report(
  *,
  candidate: dict[str, Any],
  baseline: dict[str, Any] | None,
  success_threshold: float,
) -> dict[str, Any]:
  candidate_distance = float(candidate["style"]["nearest_distance_mean"])
  baseline_distance = None if baseline is None else float(baseline["style"]["nearest_distance_mean"])
  improvement = None if baseline_distance is None else baseline_distance - candidate_distance
  candidate_success_rate = _success_rate(candidate)
  return {
    "schema_version": SCHEMA_VERSION,
    "type": "summary",
    "status": "ok",
    "candidate": candidate,
    "baseline": baseline,
    "candidate_success_rate": candidate_success_rate,
    "success_threshold": float(success_threshold),
    "style_distance_mean": candidate_distance,
    "baseline_style_distance_mean": baseline_distance,
    "style_improvement_vs_baseline": improvement,
    "style_distance_improved": bool(improvement is not None and improvement > 0.0),
    "style_gate_pass": bool(
      candidate_success_rate is not None
      and candidate_success_rate >= float(success_threshold)
      and (baseline is None or (improvement is not None and improvement > 0.0))
    ),
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", required=True, type=Path)
  parser.add_argument("--baseline-checkpoint-file", type=Path, default=None)
  parser.add_argument("--demo-data-dir", default="data/motions/g1_getup_amp")
  parser.add_argument("--manifest-path", default=None)
  parser.add_argument("--amp-target-dt", type=float, default=0.02)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=700)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--stop-on-done", action="store_true")
  parser.add_argument("--success-threshold", type=float, default=0.95)
  parser.add_argument("--output", type=Path, default=None)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    with contextlib.redirect_stdout(sys.stderr):
      dataset = _load_expert_dataset(args)
      expert_obs = dataset.amp_obs.detach().cpu()
      candidate = evaluate_checkpoint(args, args.checkpoint_file, expert_obs)
      baseline = (
        evaluate_checkpoint(args, args.baseline_checkpoint_file, expert_obs)
        if args.baseline_checkpoint_file is not None
        else None
      )
    report = build_comparison_report(
      candidate=candidate,
      baseline=baseline,
      success_threshold=args.success_threshold,
    )
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["style_gate_pass"] else 1
  except Exception as exc:
    blocker = rollout.build_blocker_record(
      task_id="Unitree-G1-GetUp-AMP",
      phase="amp_style_eval",
      exc=exc,
      request={
        "checkpoint_file": str(args.checkpoint_file),
        "baseline_checkpoint_file": str(args.baseline_checkpoint_file) if args.baseline_checkpoint_file else None,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "device": args.device,
      },
    )
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(blocker, indent=2, sort_keys=True))
    print(json.dumps(blocker, sort_keys=True))
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
