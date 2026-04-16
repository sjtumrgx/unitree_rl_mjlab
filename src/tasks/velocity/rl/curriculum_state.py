"""State helpers for the anti-fall curriculum runner."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg

from .antifall_curriculum import curriculum_stage_name


@dataclass(frozen=True)
class ObservationGroupSignature:
  terms: tuple[str, ...]
  history_length: int


@dataclass(frozen=True)
class StageObservationSignature:
  actor: ObservationGroupSignature
  critic: ObservationGroupSignature


def stage_observation_signature(task_id: str) -> StageObservationSignature:
  cfg = load_env_cfg(task_id)
  actor_group = cfg.observations["actor"]
  critic_group = cfg.observations["critic"]
  return StageObservationSignature(
    actor=ObservationGroupSignature(
      terms=tuple(actor_group.terms),
      history_length=actor_group.history_length,
    ),
    critic=ObservationGroupSignature(
      terms=tuple(critic_group.terms),
      history_length=critic_group.history_length,
    ),
  )


def transition_load_mode(source_task_id: str, target_task_id: str) -> str:
  source_sig = stage_observation_signature(source_task_id)
  target_sig = stage_observation_signature(target_task_id)
  if source_sig.actor != target_sig.actor:
    return "fresh"
  if source_sig.critic != target_sig.critic:
    return "actor_only"
  return "full"


def new_curriculum_manifest(
  *,
  curriculum_task_id: str,
  stage_task_ids: tuple[str, ...],
  started_at: str,
) -> dict[str, Any]:
  return {
    "curriculum_task_id": curriculum_task_id,
    "started_at": started_at,
    "updated_at": started_at,
    "status": "starting",
    "active_stage_index": 0,
    "completed_stage_indices": [],
    "latest_checkpoint": None,
    "failure_reason": None,
    "stages": [
      {
        "task_id": task_id,
        "stage_name": curriculum_stage_name(task_id),
        "status": "pending",
        "load_mode": None,
        "promotion_reason": None,
        "stage_iteration": 0,
        "global_iteration": 0,
        "source_checkpoint": None,
        "latest_checkpoint": None,
        "actor_signature": asdict(stage_observation_signature(task_id).actor),
        "critic_signature": asdict(stage_observation_signature(task_id).critic),
        "metrics": {},
        "started_at": None,
        "completed_at": None,
      }
      for task_id in stage_task_ids
    ],
  }


def load_manifest(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def manifest_for_update(manifest: dict[str, Any]) -> dict[str, Any]:
  return deepcopy(manifest)
