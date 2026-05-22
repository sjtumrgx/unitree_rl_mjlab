"""Run bounded AntiFall-GetUp rollout telemetry.

Acceptance is intentionally stricter than "does not crash": a rollout must show
walking command tracking, at least one disturbance/fallen phase, recovery events,
and resumed controllable locomotion after the disturbance window.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "g1-antifall-getup-rollout-v1"
TASK_ID = "Unitree-G1-AntiFall-GetUp"


def _scalar(value: Any, default: float | None = None) -> float | None:
  if value is None:
    return default
  if torch.is_tensor(value):
    if value.numel() == 0:
      return default
    value = value.detach().flatten()[0].cpu().item()
  try:
    out = float(value)
  except (TypeError, ValueError):
    return default
  return out if math.isfinite(out) else default


def _scene_get(scene: Any, name: str, default: Any = None) -> Any:
  if name == "env_origins" and hasattr(scene, "env_origins"):
    return scene.env_origins
  if isinstance(scene, dict):
    return scene.get(name, default)
  try:
    return scene[name]
  except Exception:
    return default


def _body_id(asset: Any, body_name: str) -> int | None:
  names = getattr(asset, "body_names", None)
  if names is not None and body_name in names:
    return int(list(names).index(body_name))
  finder = getattr(asset, "find_bodies", None)
  if callable(finder):
    try:
      ids, _ = finder(body_name)
      return int(ids[0]) if ids else None
    except Exception:
      return None
  return None


def _relative_z(env: Any, z: torch.Tensor) -> torch.Tensor:
  origins = _scene_get(env.scene, "env_origins")
  if origins is None:
    return z
  return z - origins[:, 2]


def _metric_terms(env: Any) -> dict[str, float]:
  manager = getattr(env, "metrics_manager", None)
  names = list(getattr(manager, "_term_names", ()))
  values = getattr(manager, "_step_values", None)
  if values is None:
    return {}
  out: dict[str, float] = {}
  for idx, name in enumerate(names):
    value = _scalar(values[:, idx].mean())
    if value is not None:
      out[str(name)] = float(value)
  return out


def _command_stats(env: Any, asset: Any, command_name: str = "twist") -> dict[str, float | None]:
  command_manager = getattr(env, "command_manager", None)
  command = command_manager.get_command(command_name) if command_manager is not None else None
  if command is None:
    return {
      "moving_command_rate": None,
      "tracking_rate": None,
      "lin_error_mean": None,
      "yaw_error_mean": None,
    }
  lin_vel_b = getattr(asset.data, "root_link_lin_vel_b", None)
  ang_vel_b = getattr(asset.data, "root_link_ang_vel_b", None)
  if lin_vel_b is None or ang_vel_b is None:
    return {
      "moving_command_rate": None,
      "tracking_rate": None,
      "lin_error_mean": None,
      "yaw_error_mean": None,
    }
  lin_error = torch.linalg.norm(command[:, :2] - lin_vel_b[:, :2], dim=1)
  yaw_error = torch.abs(command[:, 2] - ang_vel_b[:, 2])
  moving = (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > 0.1
  tracking = (lin_error <= 0.5) & (yaw_error <= 0.75)
  return {
    "moving_command_rate": _scalar(moving.float().mean()),
    "tracking_rate": _scalar((tracking & moving).float().mean()),
    "lin_error_mean": _scalar(lin_error.mean()),
    "yaw_error_mean": _scalar(yaw_error.mean()),
  }


def _root_stats(env: Any, asset: Any) -> dict[str, float | None]:
  root_z = _relative_z(env, asset.data.root_link_pos_w[:, 2])
  torso_id = _body_id(asset, "torso_link")
  if torso_id is not None and hasattr(asset.data, "body_link_pos_w"):
    torso_height = _relative_z(env, asset.data.body_link_pos_w[:, torso_id, 2])
  else:
    torso_height = root_z
  tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  fallen = (torso_height < 0.35) | (tilt > 0.75)
  return {
    "root_z_mean": _scalar(root_z.mean()),
    "torso_height_mean": _scalar(torso_height.mean()),
    "tilt_mean": _scalar(tilt.mean()),
    "fallen_rate": _scalar(fallen.float().mean()),
  }


def build_step_record(env: Any, *, step_index: int, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, Any]) -> dict[str, Any]:
  asset = _scene_get(env.scene, "robot")
  return {
    "schema_version": SCHEMA_VERSION,
    "type": "step",
    "status": "ok",
    "task_id": TASK_ID,
    "step": int(step_index),
    "command": _command_stats(env, asset),
    "root": _root_stats(env, asset),
    "metrics": _metric_terms(env),
    "reward_mean": _scalar(rewards.mean()),
    "done_any": bool(torch.as_tensor(dones).bool().any().item()),
    "time_out_any": bool(torch.as_tensor(extras.get("time_outs", False)).bool().any().item()),
  }


def summarize_records(records: list[dict[str, Any]], *, success_threshold: float = 0.8) -> dict[str, Any]:
  metadata = next((record for record in records if record.get("type") == "metadata"), {})
  step_records = [record for record in records if record.get("type") == "step"]
  num_envs = int(metadata.get("num_envs") or 0)
  disturbance_events_per_env = sum(float(r.get("metrics", {}).get("disturbance_count") or 0.0) for r in step_records)
  recovery_events_per_env = sum(float(r.get("metrics", {}).get("recovery_success_count") or 0.0) for r in step_records)
  controllable_rates = [float(r.get("metrics", {}).get("controllable_locomotion") or 0.0) for r in step_records]
  tracking_rates = [float(r.get("command", {}).get("tracking_rate") or 0.0) for r in step_records]
  fallen_rates = [float(r.get("root", {}).get("fallen_rate") or 0.0) for r in step_records]
  recovery_latencies = [
    float(r.get("metrics", {}).get("recovery_latency") or 0.0)
    for r in step_records
    if float(r.get("metrics", {}).get("recovery_latency") or 0.0) > 0.0
  ]
  first_disturbance_index = next(
    (idx for idx, r in enumerate(step_records) if float(r.get("metrics", {}).get("disturbance_count") or 0.0) > 0.0),
    None,
  )
  pre_disturbance_tracking = max(tracking_rates[:first_disturbance_index], default=max(tracking_rates, default=0.0)) if first_disturbance_index is not None else max(tracking_rates, default=0.0)
  post_disturbance_controllable = max(controllable_rates[first_disturbance_index + 1 :], default=controllable_rates[-1] if controllable_rates else 0.0) if first_disturbance_index is not None else 0.0
  max_fallen_rate = max(fallen_rates, default=0.0)
  final_controllable_rate = controllable_rates[-1] if controllable_rates else 0.0
  gate = (
    pre_disturbance_tracking >= success_threshold
    and disturbance_events_per_env > 0.0
    and max_fallen_rate > 0.0
    and recovery_events_per_env > 0.0
    and post_disturbance_controllable >= success_threshold
  )
  return {
    "schema_version": SCHEMA_VERSION,
    "type": "summary",
    "status": "ok",
    "task_id": metadata.get("task_id", TASK_ID),
    "checkpoint_file": metadata.get("checkpoint_file"),
    "steps_recorded": len(step_records),
    "num_envs": num_envs or None,
    "success_threshold": float(success_threshold),
    "pre_disturbance_tracking_rate": pre_disturbance_tracking,
    "post_disturbance_controllable_rate": post_disturbance_controllable,
    "final_controllable_rate": final_controllable_rate,
    "max_fallen_rate": max_fallen_rate,
    "disturbance_events_per_env": disturbance_events_per_env,
    "disturbance_count_estimate": int(round(disturbance_events_per_env * num_envs)) if num_envs > 0 else None,
    "recovery_events_per_env": recovery_events_per_env,
    "recovery_success_count_estimate": int(round(recovery_events_per_env * num_envs)) if num_envs > 0 else None,
    "mean_recovery_latency_s": sum(recovery_latencies) / len(recovery_latencies) if recovery_latencies else None,
    "walk_disturb_recover_resume_gate": bool(gate),
    "risk_flags": {
      "no_disturbance_seen": disturbance_events_per_env <= 0.0,
      "no_fallen_phase_seen": max_fallen_rate <= 0.0,
      "no_recovery_success_seen": recovery_events_per_env <= 0.0,
      "final_controllable_below_threshold": final_controllable_rate < success_threshold,
    },
  }


def _make_dummy_policy(agent: str, env) -> Any:
  shape = (env.num_envs, env.num_actions)

  class _Policy:
    def __call__(self, obs):
      del obs
      if agent == "zero":
        return torch.zeros(shape, device=env.device)
      return 2.0 * torch.rand(shape, device=env.device) - 1.0

  return _Policy()


def _make_trained_policy(args: argparse.Namespace, env, agent_cfg):
  if args.checkpoint_file is None:
    raise ValueError("--checkpoint-file is required when --agent=trained")
  checkpoint = Path(args.checkpoint_file).expanduser()
  if not checkpoint.exists():
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls
  from scripts.train import _load_actor_with_compatible_input_expansion

  agent_dict = asdict(agent_cfg)
  agent_dict["logger"] = "tensorboard"
  agent_dict["upload_model"] = False
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_dict, None, args.device)
  _load_actor_with_compatible_input_expansion(
    runner,
    checkpoint,
    load_cfg={"actor": True},
    map_location=args.device,
  )
  return runner.get_inference_policy(device=args.device)


def run_rollout_records(args: argparse.Namespace) -> list[dict[str, Any]]:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.velocity.config.g1_antifall.env_cfgs import unitree_g1_antifall_getup_env_cfg

  configure_torch_backends()
  # Gate rollouts should prove the BFM-style lifecycle:
  # nominal walking first, explicit push/fall disturbance, then recovery.
  # The registered play config keeps a small hard-reset probability to preserve
  # the task's fallen-start evaluation coverage, but using it here marks a
  # near-failure reset at t=0 and poisons "pre-disturbance tracking".  Build the
  # env directly with hard resets disabled unless the caller explicitly asks for
  # train-like curriculum coverage.
  env_cfg = unitree_g1_antifall_getup_env_cfg(
    play=not bool(args.train_like),
    hard_reset_prob=None if bool(args.train_like) else 0.0,
  )
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = int(args.num_envs)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    policy = _make_trained_policy(args, env, agent_cfg) if args.agent == "trained" else _make_dummy_policy(args.agent, env)
    records: list[dict[str, Any]] = [
      {
        "schema_version": SCHEMA_VERSION,
        "type": "metadata",
        "status": "ok",
        "task_id": TASK_ID,
        "mode": "train-like" if args.train_like else "play-like",
        "agent": args.agent,
        "checkpoint_file": str(args.checkpoint_file) if args.checkpoint_file else None,
        "num_envs": int(args.num_envs),
        "steps_requested": int(args.steps),
        "clip_actions": agent_cfg.clip_actions,
      }
    ]
    obs = env.get_observations()
    for step_index in range(int(args.steps)):
      with torch.no_grad():
        action = policy(obs)
      obs, rewards, dones, extras = env.step(action)
      records.append(build_step_record(raw_env, step_index=step_index, rewards=rewards.detach(), dones=dones.detach(), extras=extras))
      if bool(args.stop_on_done) and torch.as_tensor(dones).bool().any().item():
        break
    records.append(summarize_records(records, success_threshold=args.success_threshold))
    return records
  finally:
    env.close()


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, default=None)
  parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--train-like", action="store_true")
  parser.add_argument("--stop-on-done", action="store_true")
  parser.add_argument("--success-threshold", type=float, default=0.8)
  parser.add_argument("--output", type=Path, default=None)
  return parser


def _write(output: Path | None, records: list[dict[str, Any]]) -> None:
  if output is None:
    for record in records:
      print(json.dumps(record, sort_keys=True))
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    with contextlib.redirect_stdout(sys.stderr):
      records = run_rollout_records(args)
    _write(args.output, records)
    print(json.dumps(records[-1], sort_keys=True))
    return 0 if records[-1].get("walk_disturb_recover_resume_gate") else 1
  except Exception as exc:
    record = {
      "schema_version": SCHEMA_VERSION,
      "type": "blocker",
      "status": "blocked",
      "task_id": TASK_ID,
      "blocker": {
        "exception_type": exc.__class__.__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc(limit=6).splitlines()[-12:],
      },
    }
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps(record, sort_keys=True))
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
