"""Run bounded G1 GetUp rollout telemetry or emit a structured blocker.

This helper is intentionally diagnostic-only.  It does not train, does not open
viewers, and does not alter robot assets.  When simulator/checkpoint setup is not
available, it writes a blocker JSON so the debug report can distinguish a real
telemetry result from an environment limitation.
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
from typing import Any, Literal

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

SCHEMA_VERSION = "g1-getup-rollout-v1"
GETUP_TASKS = ("Unitree-G1-GetUp", "Unitree-G1-GetUp-AMP")
_TRACKED_REWARD_TERMS = {
  "host_lift_progress",
  "host_task_reward",
  "host_target_standing",
  "host_feet_support",
  "host_action_smoothness",
  "action_rate_l2",
  "joint_acc_l2",
  "support_body_contact_penalty_after_lift",
  "pelvis_clearance_penalty",
}


def _scalar(value: Any, default: float | bool | None = None) -> float | bool | None:
  if value is None:
    return default
  if torch.is_tensor(value):
    if value.numel() == 0:
      return default
    value = value.detach().flatten()[0].cpu().item()
  if isinstance(value, bool):
    return value
  try:
    out = float(value)
  except (TypeError, ValueError):
    return default
  if not math.isfinite(out):
    return default
  return out


def _max_abs(value: torch.Tensor | None) -> float | None:
  if value is None or value.numel() == 0:
    return None
  return _scalar(torch.max(torch.abs(value.detach())))  # type: ignore[return-value]


def _l2_max(value: torch.Tensor | None) -> float | None:
  if value is None or value.numel() == 0:
    return None
  flat = value.detach().flatten(start_dim=1) if value.ndim > 1 else value.detach().reshape(1, -1)
  return _scalar(torch.linalg.norm(flat, dim=1).amax())  # type: ignore[return-value]


def _scene_get(scene: Any, name: str, default: Any = None) -> Any:
  if name == "env_origins" and hasattr(scene, "env_origins"):
    return scene.env_origins
  if isinstance(scene, dict):
    return scene.get(name, default)
  try:
    return scene[name]
  except Exception:
    return default


def _relative_z(env: Any, z: torch.Tensor) -> torch.Tensor:
  origins = _scene_get(env.scene, "env_origins")
  if origins is None:
    return z
  return z - origins[:, 2]


def _body_id(asset: Any, body_name: str) -> int | None:
  body_names = getattr(asset, "body_names", None)
  if body_names is not None and body_name in body_names:
    return int(list(body_names).index(body_name))
  find_bodies = getattr(asset, "find_bodies", None)
  if callable(find_bodies):
    try:
      ids, _ = find_bodies(body_name)
      if ids:
        return int(ids[0])
    except Exception:
      return None
  return None


def _contact_count(env: Any, sensor_name: str) -> float | None:
  sensor = _scene_get(env.scene, sensor_name)
  found = getattr(getattr(sensor, "data", None), "found", None)
  if found is None:
    return None
  return _scalar((found > 0).float().flatten(start_dim=1).sum(dim=1).amax())


def _reward_terms(env: Any) -> dict[str, float]:
  manager = getattr(env, "reward_manager", None)
  names = list(getattr(manager, "_term_names", ()))
  values = getattr(manager, "_step_reward", None)
  if values is None:
    return {}
  out: dict[str, float] = {}
  for idx, name in enumerate(names):
    if name not in _TRACKED_REWARD_TERMS:
      continue
    value = _scalar(values[:, idx].mean())
    if value is not None:
      out[name] = float(value)
  return out


def _termination_terms(env: Any) -> dict[str, bool]:
  manager = getattr(env, "termination_manager", None)
  dones = getattr(manager, "_term_dones", None)
  if not isinstance(dones, dict):
    return {}
  return {str(name): bool(torch.as_tensor(value).any().item()) for name, value in dones.items()}


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


def _action_term_tensors(env: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  action_manager = getattr(env, "action_manager", None)
  terms = getattr(action_manager, "_terms", {})
  term = terms.get("joint_pos") if isinstance(terms, dict) else None
  processed = getattr(term, "_processed_actions", None)
  raw = getattr(term, "_raw_actions", None)
  return processed, raw


def _joint_target_delta(env: Any, asset: Any) -> dict[str, float | None]:
  written_delta = getattr(env, "_host_getup_joint_position_delta", None)
  target = getattr(env, "_host_getup_joint_position_target", None)
  if target is None:
    return {"joint_target_delta_max": None, "joint_target_abs_max": None}
  if written_delta is not None:
    return {
      "joint_target_delta_max": _max_abs(written_delta),
      "joint_target_abs_max": _max_abs(target),
    }
  target_ids = getattr(env, "_host_getup_joint_target_ids", None)
  if target_ids is None:
    current = asset.data.joint_pos[:, : target.shape[1]]
  else:
    current = asset.data.joint_pos[:, target_ids]
  delta = target - current
  return {
    "joint_target_delta_max": _max_abs(delta),
    "joint_target_abs_max": _max_abs(target),
  }


def _root_dynamics(env: Any, asset: Any) -> dict[str, float | None]:
  root_z = _relative_z(env, asset.data.root_link_pos_w[:, 2])
  torso_id = _body_id(asset, "torso_link")
  if torso_id is None:
    torso_height = root_z
  else:
    torso_height = _relative_z(env, asset.data.body_link_pos_w[:, torso_id, 2])
  projected_gravity = asset.data.projected_gravity_b
  return {
    "root_z": _scalar(root_z),
    "root_vertical_velocity": _scalar(asset.data.root_link_lin_vel_w[:, 2].amax()),
    "root_lin_vel_norm": _l2_max(asset.data.root_link_lin_vel_w),
    "root_ang_vel_norm": _l2_max(asset.data.root_link_ang_vel_w),
    "torso_height": _scalar(torso_height),
    "upright_alignment": _scalar((-projected_gravity[:, 2]).amax()),
    "projected_gravity_z": _scalar(projected_gravity[:, 2].mean()),
  }


def _curriculum_state(env: Any, train_like: bool) -> dict[str, float | bool | None]:
  state = getattr(env, "_host_getup_curriculum_state", None)
  return {
    "train_like": bool(train_like),
    "assist_event_active": "getup_assist_force" in getattr(getattr(env, "cfg", None), "events", {}),
    "getup_assist_force_n": _scalar(state.get("force_n").amax()) if isinstance(state, dict) else None,
    "getup_action_rescale": _scalar(state.get("action_rescale").amax()) if isinstance(state, dict) else None,
    "max_torso_height": _scalar(state.get("max_torso_height").amax()) if isinstance(state, dict) else None,
  }


def build_step_record(
  env: Any,
  *,
  task_id: str,
  step_index: int,
  mode: Literal["train-like", "play-like"],
  raw_action: torch.Tensor,
  clipped_action: torch.Tensor,
  previous_clipped_action: torch.Tensor | None,
  rewards: torch.Tensor,
  dones: torch.Tensor,
  extras: dict[str, Any],
  clip_actions: float | None,
  amp_stats: dict[str, float | str | None] | None = None,
) -> dict[str, Any]:
  """Build one JSON-serializable telemetry row from a live or fake env."""
  asset = _scene_get(env.scene, "robot")
  processed_actions, term_raw_actions = _action_term_tensors(env)
  action_rate = None
  if previous_clipped_action is not None:
    action_rate = clipped_action - previous_clipped_action

  record = {
    "schema_version": SCHEMA_VERSION,
    "type": "step",
    "status": "ok",
    "task_id": task_id,
    "mode": mode,
    "step": int(step_index),
    "action": {
      "clip_actions": clip_actions,
      "raw_max_abs": _max_abs(raw_action),
      "raw_l2_max": _l2_max(raw_action),
      "clipped_max_abs": _max_abs(clipped_action),
      "clipped_l2_max": _l2_max(clipped_action),
      "action_rate_max_abs": _max_abs(action_rate),
      "term_raw_max_abs": _max_abs(term_raw_actions),
      "processed_max_abs": _max_abs(processed_actions),
      "processed_l2_max": _l2_max(processed_actions),
    },
    "target": _joint_target_delta(env, asset),
    "root": _root_dynamics(env, asset),
    "curriculum_assist": _curriculum_state(env, train_like=mode == "train-like"),
    "support": {
      "feet_contact_count": _contact_count(env, "feet_ground_contact"),
      "support_body_contact_count": _contact_count(env, "support_body_contact"),
    },
    "reward": {
      "total_mean": _scalar(rewards.mean()),
      "terms": _reward_terms(env),
    },
    "termination": {
      "done_any": bool(torch.as_tensor(dones).bool().any().item()),
      "terms": _termination_terms(env),
      "time_out_any": bool(torch.as_tensor(extras.get("time_outs", False)).bool().any().item()),
    },
    "metrics": _metric_terms(env),
  }
  if amp_stats is not None:
    record["amp"] = amp_stats
  return record


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
  step_records = [record for record in records if record.get("type") == "step"]
  max_target_delta = max(
    (float(record["target"]["joint_target_delta_max"] or 0.0) for record in step_records),
    default=0.0,
  )
  max_upward_velocity = max(
    (max(float(record["root"]["root_vertical_velocity"] or 0.0), 0.0) for record in step_records),
    default=0.0,
  )
  max_vertical_speed = max(
    (abs(float(record["root"]["root_vertical_velocity"] or 0.0)) for record in step_records),
    default=0.0,
  )
  max_torso_height = max(
    (float(record["root"]["torso_height"] or 0.0) for record in step_records),
    default=0.0,
  )
  first_done_step = next(
    (int(record["step"]) for record in step_records if record["termination"]["done_any"]),
    None,
  )
  supportless_height_spike = any(
    float(record["root"]["torso_height"] or 0.0) > 0.9
    and float(record["support"]["feet_contact_count"] or 0.0) < 1.0
    for record in step_records
  )
  return {
    "schema_version": SCHEMA_VERSION,
    "type": "summary",
    "status": "ok",
    "steps_recorded": len(step_records),
    "first_done_step": first_done_step,
    "max_joint_target_delta": max_target_delta,
    "max_root_upward_velocity": max_upward_velocity,
    "max_root_vertical_speed": max_vertical_speed,
    # Backwards-compatible alias for earlier reports/tests.  This is speed
    # (absolute vertical velocity), not signed upward velocity.
    "max_root_vertical_velocity": max_vertical_speed,
    "max_torso_height": max_torso_height,
    "risk_flags": {
      "target_delta_gt_1rad": max_target_delta > 1.0,
      "upward_velocity_gt_2mps": max_upward_velocity > 2.0,
      "vertical_speed_gt_2mps": max_vertical_speed > 2.0,
      # Backwards-compatible alias used by older diagnostics.
      "vertical_velocity_gt_2mps": max_vertical_speed > 2.0,
      "supportless_height_spike": supportless_height_spike,
    },
  }


def build_blocker_record(
  *,
  task_id: str,
  phase: str,
  exc: BaseException,
  request: dict[str, Any],
) -> dict[str, Any]:
  return {
    "schema_version": SCHEMA_VERSION,
    "type": "blocker",
    "status": "blocked",
    "task_id": task_id,
    "telemetry_required": True,
    "request": request,
    "blocker": {
      "phase": phase,
      "exception_type": exc.__class__.__name__,
      "message": str(exc),
      "traceback_tail": traceback.format_exc(limit=6).splitlines()[-12:],
    },
  }


def _write_json_or_jsonl(output: Path | None, records: list[dict[str, Any]]) -> None:
  if output is None:
    for record in records:
      print(json.dumps(record, sort_keys=True))
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  if len(records) == 1 and records[0].get("type") == "blocker":
    output.write_text(json.dumps(records[0], indent=2, sort_keys=True))
    return
  output.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")


def _load_getup_configs(args: argparse.Namespace):
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from src.tasks.velocity.config.g1_getup.env_cfgs import (
    unitree_g1_getup_amp_env_cfg,
    unitree_g1_getup_env_cfg,
  )
  from src.tasks.velocity.config.g1_getup.rl_cfg import (
    unitree_g1_getup_amp_ppo_runner_cfg,
    unitree_g1_getup_ppo_runner_cfg,
  )

  play = not bool(args.train_like)
  if args.task_id == "Unitree-G1-GetUp":
    return (
      unitree_g1_getup_env_cfg(terrain=args.getup_terrain, play=play),
      unitree_g1_getup_ppo_runner_cfg(terrain=args.getup_terrain),
    )
  if args.task_id == "Unitree-G1-GetUp-AMP":
    return (
      unitree_g1_getup_amp_env_cfg(demo_data_dir=args.demo_data_dir, play=play),
      unitree_g1_getup_amp_ppo_runner_cfg(
        demo_data_dir=args.demo_data_dir,
        manifest_path=args.manifest_path,
      ),
    )
  return load_env_cfg(args.task_id, play=play), load_rl_cfg(args.task_id)


def _make_dummy_policy(agent: str, env) -> Any:
  action_shape = (env.num_envs, env.num_actions)

  class _Policy:
    def __call__(self, obs):
      del obs
      if agent == "zero":
        return torch.zeros(action_shape, device=env.device)
      return 2.0 * torch.rand(action_shape, device=env.device) - 1.0

  return _Policy()


def _make_trained_policy(args: argparse.Namespace, env, agent_cfg):
  if args.checkpoint_file is None:
    raise ValueError("--checkpoint-file is required when --agent=trained")
  checkpoint = Path(args.checkpoint_file).expanduser()
  if not checkpoint.exists():
    raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_runner_cls

  agent_dict = asdict(agent_cfg)
  agent_dict["logger"] = "tensorboard"
  agent_dict["upload_model"] = False
  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_dict, None, args.device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
  return runner.get_inference_policy(device=args.device), runner


def _amp_step_stats(runner: Any, prev_obs: Any, next_obs: Any) -> dict[str, float | str | None] | None:
  alg = getattr(runner, "alg", None)
  if alg is None or not hasattr(alg, "_amp_transition"):
    return None
  try:
    with torch.no_grad():
      transitions = alg._amp_transition(prev_obs, next_obs)
      logits = alg.discriminator(transitions)
      reward = alg.discriminator.reward(transitions)
  except Exception:
    return {
      "amp_error": "failed_to_compute_amp_stats",
      "manifest_path": str(getattr(alg, "manifest_path", "")) or None,
    }
  return {
    "obs_dim": float(getattr(alg, "amp_obs_dim", transitions.shape[-1] // 2)),
    "reward_mean": _scalar(reward.mean()),
    "policy_score": _scalar(torch.sigmoid(logits).mean()),
    "manifest_path": str(getattr(alg, "manifest_path", "")) or None,
  }


def _run_rollout_records(args: argparse.Namespace) -> list[dict[str, Any]]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  env_cfg, agent_cfg = _load_getup_configs(args)
  env_cfg.scene.num_envs = int(args.num_envs)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner = None
  try:
    if args.agent == "trained":
      policy, runner = _make_trained_policy(args, env, agent_cfg)
    else:
      policy = _make_dummy_policy(args.agent, env)

    mode = "train-like" if args.train_like else "play-like"
    metadata = {
      "schema_version": SCHEMA_VERSION,
      "type": "metadata",
      "status": "ok",
      "task_id": args.task_id,
      "mode": mode,
      "agent": args.agent,
      "checkpoint_file": str(args.checkpoint_file) if args.checkpoint_file else None,
      "num_envs": int(args.num_envs),
      "steps_requested": int(args.steps),
      "clip_actions": agent_cfg.clip_actions,
    }
    records: list[dict[str, Any]] = [metadata]
    obs = env.get_observations()
    previous_clipped_action: torch.Tensor | None = None
    for step_index in range(int(args.steps)):
      with torch.no_grad():
        raw_action = policy(obs)
      clipped = (
        torch.clamp(raw_action, -agent_cfg.clip_actions, agent_cfg.clip_actions)
        if agent_cfg.clip_actions is not None
        else raw_action
      )
      next_obs, rewards, dones, extras = env.step(raw_action)
      amp_stats = _amp_step_stats(runner, obs, next_obs) if runner is not None else None
      records.append(
        build_step_record(
          raw_env,
          task_id=args.task_id,
          step_index=step_index,
          mode=mode,  # type: ignore[arg-type]
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
    records.append(summarize_records(records))
    return records
  finally:
    env.close()


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id", help="Task id, usually Unitree-G1-GetUp or Unitree-G1-GetUp-AMP")
  parser.add_argument("--getup-terrain", default="ground")
  parser.add_argument("--demo-data-dir", default="data/motions/g1_getup_amp")
  parser.add_argument("--manifest-path", default=None)
  parser.add_argument("--checkpoint-file", default=None)
  parser.add_argument("--agent", choices=("trained", "zero", "random"), default="trained")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--train-like", action="store_true", help="Use train env cfg with assist event enabled.")
  parser.add_argument("--stop-on-done", action="store_true")
  parser.add_argument("--output", type=Path, default=None)
  return parser


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
  return {
    "task_id": args.task_id,
    "agent": args.agent,
    "checkpoint_file": args.checkpoint_file,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "device": args.device,
    "train_like": args.train_like,
    "output": str(args.output) if args.output else None,
  }


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    with contextlib.redirect_stdout(sys.stderr):
      records = _run_rollout_records(args)
    _write_json_or_jsonl(args.output, records)
    print(json.dumps(records[-1], sort_keys=True))
    return 0
  except Exception as exc:
    record = build_blocker_record(
      task_id=args.task_id,
      phase="rollout",
      exc=exc,
      request=_request_from_args(args),
    )
    _write_json_or_jsonl(args.output, [record])
    print(json.dumps(record, sort_keys=True))
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
