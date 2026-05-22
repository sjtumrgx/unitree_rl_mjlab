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
  "host_upright_progress",
  "host_support_relief",
  "host_task_reward",
  "host_target_standing",
  "host_feet_support",
  "getup_completion_bonus",
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


def _tensor_stat(value: torch.Tensor | None, reducer: str) -> float | None:
  if value is None or value.numel() == 0:
    return None
  tensor = value.detach().float()
  if reducer == "min":
    return _scalar(tensor.amin())  # type: ignore[return-value]
  if reducer == "mean":
    return _scalar(tensor.mean())  # type: ignore[return-value]
  return _scalar(tensor.amax())  # type: ignore[return-value]


def _contact_count_tensor(env: Any, sensor_name: str) -> torch.Tensor | None:
  sensor = _scene_get(env.scene, sensor_name)
  found = getattr(getattr(sensor, "data", None), "found", None)
  if found is None:
    return None
  return (found > 0).float().flatten(start_dim=1).sum(dim=1)


def _contact_count(env: Any, sensor_name: str) -> float | None:
  return _tensor_stat(_contact_count_tensor(env, sensor_name), "max")


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


def _metric_tensor(env: Any, metric_name: str) -> torch.Tensor | None:
  manager = getattr(env, "metrics_manager", None)
  names = list(getattr(manager, "_term_names", ()))
  values = getattr(manager, "_step_values", None)
  if values is None or metric_name not in names:
    return None
  return values[:, names.index(metric_name)].detach().float()


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


def _torso_height_tensor(env: Any, asset: Any) -> torch.Tensor:
  root_z = _relative_z(env, asset.data.root_link_pos_w[:, 2])
  torso_id = _body_id(asset, "torso_link")
  if torso_id is None:
    return root_z
  return _relative_z(env, asset.data.body_link_pos_w[:, torso_id, 2])


def _supportless_height_spike(
  env: Any,
  asset: Any,
  *,
  torso_height_threshold: float = 0.9,
  min_feet_contact_count: float = 1.0,
) -> bool:
  torso_height = _torso_height_tensor(env, asset)
  feet_count = _contact_count_tensor(env, "feet_ground_contact")
  if feet_count is None:
    return False
  return bool(((torso_height > torso_height_threshold) & (feet_count < min_feet_contact_count)).any().item())


def _root_dynamics(env: Any, asset: Any) -> dict[str, float | None]:
  root_z = _relative_z(env, asset.data.root_link_pos_w[:, 2])
  torso_height = _torso_height_tensor(env, asset)
  projected_gravity = asset.data.projected_gravity_b
  upright_alignment = -projected_gravity[:, 2]
  return {
    # Keep the legacy singular fields as batch maxima.  Rollout diagnostics run
    # with many envs, and using element 0 here hid successful/unsafe envs in the
    # summary even though metrics were already batch means.
    "root_z": _tensor_stat(root_z, "max"),
    "root_z_min": _tensor_stat(root_z, "min"),
    "root_z_mean": _tensor_stat(root_z, "mean"),
    "root_z_max": _tensor_stat(root_z, "max"),
    "root_vertical_velocity": _scalar(asset.data.root_link_lin_vel_w[:, 2].amax()),
    "root_lin_vel_norm": _l2_max(asset.data.root_link_lin_vel_w),
    "root_ang_vel_norm": _l2_max(asset.data.root_link_ang_vel_w),
    "torso_height": _tensor_stat(torso_height, "max"),
    "torso_height_min": _tensor_stat(torso_height, "min"),
    "torso_height_mean": _tensor_stat(torso_height, "mean"),
    "torso_height_max": _tensor_stat(torso_height, "max"),
    "upright_alignment": _tensor_stat(upright_alignment, "max"),
    "upright_alignment_min": _tensor_stat(upright_alignment, "min"),
    "upright_alignment_mean": _tensor_stat(upright_alignment, "mean"),
    "upright_alignment_max": _tensor_stat(upright_alignment, "max"),
    "projected_gravity_z": _scalar(projected_gravity[:, 2].mean()),
    "supportless_height_spike": _supportless_height_spike(env, asset),
  }


def _curriculum_state(env: Any, train_like: bool) -> dict[str, float | bool | None]:
  state = getattr(env, "_host_getup_curriculum_state", None)
  force = state.get("force_n") if isinstance(state, dict) else None
  action_rescale = state.get("action_rescale") if isinstance(state, dict) else None
  episode_success = state.get("episode_success") if isinstance(state, dict) else None
  episode_force_scale = state.get("episode_force_scale") if isinstance(state, dict) else None
  return {
    "train_like": bool(train_like),
    "assist_event_active": "getup_assist_force" in getattr(getattr(env, "cfg", None), "events", {}),
    "getup_assist_force_n": _tensor_stat(force, "max"),
    "getup_assist_force_n_min": _tensor_stat(force, "min"),
    "getup_assist_force_n_mean": _tensor_stat(force, "mean"),
    "getup_action_rescale": _tensor_stat(action_rescale, "max"),
    "getup_action_rescale_min": _tensor_stat(action_rescale, "min"),
    "getup_action_rescale_mean": _tensor_stat(action_rescale, "mean"),
    "episode_success_latched_rate": _tensor_stat(episode_success.float(), "mean") if torch.is_tensor(episode_success) else None,
    "episode_force_scale_min": _tensor_stat(episode_force_scale, "min"),
    "episode_force_scale_mean": _tensor_stat(episode_force_scale, "mean"),
    "episode_force_scale_max": _tensor_stat(episode_force_scale, "max"),
    "max_torso_height": _scalar(state.get("max_torso_height").amax()) if isinstance(state, dict) else None,
  }


def _assist_success_split(env: Any) -> dict[str, dict[str, float | int]] | None:
  """Return per-step success/upright counts split by per-env assist mask.

  ``episode_force_scale_mean`` is useful telemetry, but it cannot tell whether
  the successful envs in a mixed batch were assisted or no-assist.  Keep this
  compact (counts, not arrays) so JSONL remains small while train-like
  diagnostics can distinguish real play-transfer progress from assist-only
  success.
  """

  state = getattr(env, "_host_getup_curriculum_state", None)
  if not isinstance(state, dict):
    return None
  episode_force_scale = state.get("episode_force_scale")
  if episode_force_scale is None or not torch.is_tensor(episode_force_scale):
    return None

  success = _metric_tensor(env, "getup_success_count")
  upright = _metric_tensor(env, "getup_upright")
  if success is None and upright is None:
    return None
  scale = episode_force_scale.detach().float()
  if (success is not None and success.shape[0] != scale.shape[0]) or (
    upright is not None and upright.shape[0] != scale.shape[0]
  ):
    return None
  assisted_mask = scale > 0.0
  no_assist_mask = ~assisted_mask

  def _group(mask: torch.Tensor) -> dict[str, float | int]:
    env_count = int(mask.sum().item())
    if env_count == 0:
      return {
        "env_count": 0,
        "success_events": 0.0,
        "upright_count": 0.0,
        "upright_rate": 0.0,
      }
    success_events = float(success[mask].sum().item()) if success is not None else 0.0
    upright_count = float(upright[mask].sum().item()) if upright is not None else 0.0
    return {
      "env_count": env_count,
      "success_events": success_events,
      "upright_count": upright_count,
      "upright_rate": upright_count / env_count,
    }

  return {
    "assisted": _group(assisted_mask),
    "no_assist": _group(no_assist_mask),
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
    "assist_success_split": _assist_success_split(env),
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
  metadata = next((record for record in records if record.get("type") == "metadata"), {})
  num_envs = int(metadata.get("num_envs") or 0)
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
    bool(record["root"].get("supportless_height_spike", False))
    or (
      float(record["root"]["torso_height"] or 0.0) > 0.9
      and float(record["support"]["feet_contact_count"] or 0.0) < 1.0
    )
    for record in step_records
  )
  success_events_per_env = sum(
    float(record.get("metrics", {}).get("getup_success_count") or 0.0)
    for record in step_records
  )
  success_count_estimate = (
    int(round(success_events_per_env * num_envs)) if num_envs > 0 else None
  )
  single_episode_success_rate = (
    max(0.0, min(1.0, success_events_per_env)) if num_envs > 0 else None
  )
  upright_rates = [
    float(record.get("metrics", {}).get("getup_upright") or 0.0)
    for record in step_records
  ]

  def _success_group_from_split(group_name: str) -> dict[str, Any] | None:
    group_steps = [
      record.get("assist_success_split", {}).get(group_name)
      for record in step_records
      if isinstance(record.get("assist_success_split"), dict)
      and isinstance(record.get("assist_success_split", {}).get(group_name), dict)
    ]
    if not group_steps:
      return None
    env_count = max(int(step.get("env_count") or 0) for step in group_steps)
    success_events = sum(float(step.get("success_events") or 0.0) for step in group_steps)
    upright_rates = [float(step.get("upright_rate") or 0.0) for step in group_steps]
    final_upright_rate = upright_rates[-1] if upright_rates else 0.0
    return {
      "records": len(group_steps),
      "env_count": env_count,
      "success_events": success_events,
      "success_events_per_env": success_events / env_count if env_count > 0 else 0.0,
      "success_count_estimate": int(round(success_events)) if env_count > 0 else None,
      "single_episode_success_rate": (
        max(0.0, min(1.0, success_events / env_count)) if env_count > 0 else 0.0
      ),
      "max_getup_upright_rate": max(upright_rates, default=0.0),
      "final_getup_upright_rate": final_upright_rate,
    }

  def _success_group_from_record_means(group_records: list[dict[str, Any]]) -> dict[str, Any]:
    events_per_env = sum(
      float(record.get("metrics", {}).get("getup_success_count") or 0.0)
      for record in group_records
    )
    group_upright_rates = [
      float(record.get("metrics", {}).get("getup_upright") or 0.0)
      for record in group_records
    ]
    return {
      "records": len(group_records),
      "env_count": num_envs or 0,
      "success_events_per_env": events_per_env,
      "success_count_estimate": int(round(events_per_env * num_envs)) if num_envs > 0 else None,
      "single_episode_success_rate": max(0.0, min(1.0, events_per_env)) if num_envs > 0 else None,
      "max_getup_upright_rate": max(group_upright_rates, default=0.0),
      "final_getup_upright_rate": group_upright_rates[-1] if group_upright_rates else 0.0,
    }

  assisted_records = [
    record
    for record in step_records
    if float(record.get("curriculum_assist", {}).get("episode_force_scale_mean") or 0.0) > 0.0
  ]
  no_assist_records = [
    record
    for record in step_records
    if (
      record.get("curriculum_assist", {}).get("episode_force_scale_mean") is not None
      and float(record.get("curriculum_assist", {}).get("episode_force_scale_mean") or 0.0) <= 0.0
    )
  ]
  total_success = {
    "num_envs": num_envs or None,
    "success_events_per_env": success_events_per_env,
    "success_count_estimate": success_count_estimate,
    "single_episode_success_rate": single_episode_success_rate,
    "max_getup_upright_rate": max(upright_rates, default=0.0),
    "final_getup_upright_rate": upright_rates[-1] if upright_rates else 0.0,
  }
  assisted_success = _success_group_from_split("assisted") or _success_group_from_record_means(assisted_records)
  no_assist_success = _success_group_from_split("no_assist") or _success_group_from_record_means(no_assist_records)

  # Play-like diagnostics have no getup_assist_force event, so every env is by
  # definition no-assist.  Mirroring the total success here prevents the summary
  # from showing a misleading no_assist=0/num_envs next to a high play success.
  has_assist_telemetry = any(
    isinstance(record.get("assist_success_split"), dict)
    or record.get("curriculum_assist", {}).get("episode_force_scale_mean") is not None
    for record in step_records
  )
  if not has_assist_telemetry:
    assisted_success = {
      "records": 0,
      "env_count": 0,
      "success_events_per_env": 0.0,
      "success_count_estimate": 0,
      "single_episode_success_rate": 0.0,
      "max_getup_upright_rate": 0.0,
      "final_getup_upright_rate": 0.0,
    }
    no_assist_success = {
      "records": len(step_records),
      "env_count": num_envs or 0,
      "success_events_per_env": success_events_per_env,
      "success_count_estimate": success_count_estimate,
      "single_episode_success_rate": single_episode_success_rate,
      "max_getup_upright_rate": max(upright_rates, default=0.0),
      "final_getup_upright_rate": upright_rates[-1] if upright_rates else 0.0,
    }
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
    "success": total_success,
    "success_by_assist": {
      "assisted": assisted_success,
      "no_assist": no_assist_success,
    },
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
  from scripts.train import _load_actor_with_compatible_input_expansion

  _load_actor_with_compatible_input_expansion(
    runner,
    checkpoint,
    load_cfg={"actor": True},
    map_location=args.device,
  )
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
