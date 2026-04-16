from pathlib import Path

from src.tasks.velocity.rl.antifall_curriculum import (
  ANTI_FALL_STAGE_TASK_IDS,
  CURRICULUM_TASK_ID,
)
from src.tasks.velocity.rl.curriculum_state import load_manifest, new_curriculum_manifest, write_manifest


REQUIRED_STAGE_KEYS = {
  "task_id",
  "stage_name",
  "status",
  "load_mode",
  "promotion_reason",
  "stage_iteration",
  "global_iteration",
  "source_checkpoint",
  "latest_checkpoint",
  "actor_signature",
  "critic_signature",
  "metrics",
  "started_at",
  "completed_at",
}


def test_manifest_schema_includes_required_top_level_and_stage_fields(tmp_path: Path) -> None:
  manifest = new_curriculum_manifest(
    curriculum_task_id=CURRICULUM_TASK_ID,
    stage_task_ids=ANTI_FALL_STAGE_TASK_IDS,
    started_at="2026-04-16T00:00:00Z",
  )
  path = tmp_path / "curriculum_manifest.json"
  write_manifest(path, manifest)
  payload = load_manifest(path)

  assert payload["curriculum_task_id"] == CURRICULUM_TASK_ID
  assert payload["active_stage_index"] == 0
  assert payload["completed_stage_indices"] == []
  assert payload["latest_checkpoint"] is None
  assert payload["failure_reason"] is None
  assert len(payload["stages"]) == len(ANTI_FALL_STAGE_TASK_IDS)
  assert REQUIRED_STAGE_KEYS.issubset(payload["stages"][0])
  assert payload["stages"][0]["stage_name"] == "stage0"
  assert payload["stages"][3]["critic_signature"]["terms"][-3:] == [
    "foot_contact_forces",
    "disturbance_metadata",
    "recovery_features",
  ]
  assert "height_scan" in payload["stages"][3]["critic_signature"]["terms"]
