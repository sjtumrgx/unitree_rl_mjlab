"""Emit static contract diagnostics for G1 GetUp shared no-demo/AMP paths."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.tasks.velocity.config.g1_getup.env_cfgs import (  # noqa: E402
  GETUP_TERRAIN_VARIANTS,
  unitree_g1_getup_amp_env_cfg,
  unitree_g1_getup_env_cfg,
)
from src.tasks.velocity.config.g1_getup.rl_cfg import (  # noqa: E402
  unitree_g1_getup_amp_ppo_runner_cfg,
  unitree_g1_getup_ppo_runner_cfg,
)
from src.tasks.velocity.rl.getup_amp_data import (  # noqa: E402
  AMP_OBS_DIM,
  CANONICAL_G1_23DOF_JOINT_NAMES,
  validate_amp_source_gate,
)


def _jsonable(value: Any) -> Any:
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, slice):
    return {
      "type": "slice",
      "start": _jsonable(value.start),
      "stop": _jsonable(value.stop),
      "step": _jsonable(value.step),
    }
  if dataclasses.is_dataclass(value):
    return {
      "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
      "fields": _jsonable(dataclasses.asdict(value)),
    }
  if isinstance(value, tuple):
    return [_jsonable(v) for v in value]
  if isinstance(value, list):
    return [_jsonable(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, type):
    return f"{value.__module__}.{value.__qualname__}"
  if hasattr(value, "__name__"):
    return value.__name__
  return repr(value)


def _event_params(cfg, name: str) -> dict[str, Any]:
  term = cfg.events.get(name)
  return dict(getattr(term, "params", {}) or {}) if term is not None else {}


def _reward_summary(cfg) -> dict[str, dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for name, term in cfg.rewards.items():
    out[name] = {
      "func": getattr(term.func, "__name__", str(term.func)),
      "weight": float(term.weight),
      "params": _jsonable(getattr(term, "params", {}) or {}),
    }
  return out


def build_diagnostic(*, terrain: str, demo_data_dir: str, manifest_path: str | None = None) -> dict[str, Any]:
  train_env = unitree_g1_getup_env_cfg(terrain=terrain, play=False)
  play_env = unitree_g1_getup_env_cfg(terrain=terrain, play=True)
  amp_env = unitree_g1_getup_amp_env_cfg(demo_data_dir=demo_data_dir, play=True)
  runner = unitree_g1_getup_ppo_runner_cfg(terrain=terrain)
  amp_runner = unitree_g1_getup_amp_ppo_runner_cfg(demo_data_dir=demo_data_dir, manifest_path=manifest_path)

  action_cfg = train_env.actions["joint_pos"]
  assist_params = _event_params(train_env, "getup_assist_force")
  initial_action_rescale = float(assist_params.get("initial_action_scale", 1.0))
  max_policy_delta_rad = float(runner.clip_actions or 0.0) * float(action_cfg.scale) * initial_action_rescale
  action_cfg_max_delta = getattr(action_cfg, "max_delta", None)
  max_env_delta_rad = (
    min(max_policy_delta_rad, float(action_cfg_max_delta))
    if action_cfg_max_delta is not None
    else max_policy_delta_rad
  )
  if max_env_delta_rad > 1.0:
    action_risk = "high"
  elif max_policy_delta_rad > 1.0:
    action_risk = "mitigated"
  else:
    action_risk = "low"

  train_assist = train_env.events.get("getup_assist_force")
  play_assist = play_env.events.get("getup_assist_force")
  reset_base_params = _event_params(train_env, "reset_base")
  reset_joint_params = _event_params(train_env, "reset_robot_joints")
  termination_summary = {
    name: {
      "func": getattr(term.func, "__name__", str(term.func)),
      "params": _jsonable(getattr(term, "params", {}) or {}),
    }
    for name, term in train_env.terminations.items()
  }

  source_gate: dict[str, Any]
  selected_manifest = Path(manifest_path).expanduser() if manifest_path else Path(demo_data_dir).expanduser() / "manifest.json"
  try:
    source_gate = {"status": validate_amp_source_gate(selected_manifest).get("status"), "error": None}
  except Exception as exc:  # diagnostic must report, not hide, missing local data
    source_gate = {"status": "ERROR", "error": str(exc)}

  return {
    "schema_version": "g1-getup-contracts-v1",
    "terrain": terrain,
    "action_envelope": {
      "clip_actions": runner.clip_actions,
      "action_cfg_scale": float(action_cfg.scale),
      "action_cfg_max_delta_rad": action_cfg_max_delta,
      "initial_action_rescale": initial_action_rescale,
      "max_policy_delta_rad": max_policy_delta_rad,
      "max_env_delta_rad": max_env_delta_rad,
      "risk_level": action_risk,
      "telemetry_required": action_risk == "high",
      "risk_only_not_fix_trigger": True,
      "semantics": "current_joint_pos + processed_action - encoder_bias",
    },
    "assist": {
      "train_event_present": train_assist is not None,
      "play_event_present": play_assist is not None,
      "train_play_mismatch_expected": train_assist is not None and play_assist is None,
      "metrics": [name for name in train_env.metrics if name in {"getup_assist_force_n", "getup_action_rescale"}],
      "params": _jsonable(assist_params),
      "stable_success_required": bool(assist_params.get("stable_success_required", True)),
      "initial_force_n": assist_params.get("initial_force_n"),
      "force_decay_n": assist_params.get("force_decay_n"),
      "action_scale_decay": assist_params.get("action_scale_decay"),
    },
    "reset": {
      "base_presets": _jsonable(reset_base_params.get("presets", ())),
      "preset_weight_stages": _jsonable(reset_base_params.get("preset_weight_stages", ())),
      "velocity_range": _jsonable(reset_base_params.get("velocity_range", {})),
      "joint_position_noise_range": reset_joint_params.get("position_noise_range"),
      "joint_velocity_range": reset_joint_params.get("velocity_range"),
      "fail_closed_required": True,
    },
    "terminations": termination_summary,
    "rewards": _reward_summary(train_env),
    "amp": {
      "env_enabled": bool(getattr(amp_env, "getup_amp_enabled", False)),
      "terrain": getattr(amp_env, "getup_terrain", None),
      "obs_dim_expected": AMP_OBS_DIM,
      "canonical_joint_count": len(CANONICAL_G1_23DOF_JOINT_NAMES),
      "canonical_joint_names": CANONICAL_G1_23DOF_JOINT_NAMES,
      "runner_obs_groups": _jsonable(amp_runner.obs_groups),
      "amp_obs_group": amp_runner.algorithm.amp_obs_group,
      "amp_reward_scale": amp_runner.algorithm.amp_reward_scale,
      "demo_data_dir": demo_data_dir,
      "manifest_path": str(selected_manifest),
      "source_gate": source_gate,
    },
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--terrain", choices=GETUP_TERRAIN_VARIANTS, default="ground")
  parser.add_argument("--demo-data-dir", default="data/motions/g1_getup_amp")
  parser.add_argument("--manifest-path", default=None)
  parser.add_argument("--output", type=Path, default=None)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  diagnostic = build_diagnostic(
    terrain=args.terrain,
    demo_data_dir=args.demo_data_dir,
    manifest_path=args.manifest_path,
  )
  text = json.dumps(diagnostic, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
