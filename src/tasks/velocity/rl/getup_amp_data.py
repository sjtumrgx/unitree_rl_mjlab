"""Data contracts for optional G1 GetUp AMP demonstrations.

The AMP path intentionally keeps third-party motion data outside git.  This module
normalizes small local motion files into a manifest-backed schema that the AMP
algorithm can consume deterministically in tests and in offline training jobs.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_G1_23DOF_XML = _REPO_ROOT / "src/assets/robots/unitree_g1/xmls/g1_23dof.xml"
_SCHEMA_VERSION = "g1-getup-amp-v1"
_GETUP_TAG_PATTERN = re.compile(
  r"(get.?up|stand.?up|fall|fallen|lie|lying|supine|prone|sit.?to.?stand|recovery)",
  re.I,
)
_KNOWN_29DOF_EXTRAS = {
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
}


def canonical_g1_23dof_joint_names() -> tuple[str, ...]:
  """Return the active G1 23DoF joint order from the checked-in MJCF."""
  root = ET.parse(_G1_23DOF_XML).getroot()
  return tuple(
    joint.attrib["name"]
    for joint in root.findall(".//joint")
    if "name" in joint.attrib and joint.attrib["name"] != "floating_base_joint"
  )


CANONICAL_G1_23DOF_JOINT_NAMES = canonical_g1_23dof_joint_names()
AMP_OBS_DIM = 3 + 4 + 2 * len(CANONICAL_G1_23DOF_JOINT_NAMES)


@dataclass(frozen=True)
class ProjectionResult:
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  source_joint_names: tuple[str, ...]
  canonical_joint_names: tuple[str, ...]
  projection: dict[str, Any]


@dataclass(frozen=True)
class SequenceValidationResult:
  path: str
  accepted: bool
  reason: str
  output_path: str | None = None
  metadata: dict[str, Any] | None = None


def _to_str(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, np.ndarray):
    if value.shape == ():
      return _to_str(value.item())
    return ",".join(_to_str(v) for v in value.tolist())
  return str(value)


def _to_str_tuple(value: Any) -> tuple[str, ...]:
  if value is None:
    return ()
  if isinstance(value, np.ndarray):
    if value.shape == ():
      return tuple(part.strip() for part in _to_str(value).split(",") if part.strip())
    return tuple(_to_str(v) for v in value.tolist())
  if isinstance(value, str):
    return tuple(part.strip() for part in value.split(",") if part.strip())
  return tuple(_to_str(v) for v in value)


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def project_to_canonical_23dof(
  joint_pos: np.ndarray,
  joint_vel: np.ndarray | None,
  source_joint_names: Sequence[str] | None,
  *,
  canonical_joint_names: Sequence[str] = CANONICAL_G1_23DOF_JOINT_NAMES,
) -> ProjectionResult:
  """Project source joint arrays into canonical active G1 23DoF order.

  Shape-only data is rejected because silently assuming joint order is the exact
  failure mode this contract is designed to prevent.
  """
  joint_pos = np.asarray(joint_pos, dtype=np.float32)
  if joint_pos.ndim != 2:
    raise ValueError(f"joint_pos must be [T, J], got shape {joint_pos.shape}")
  if joint_vel is None:
    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
  else:
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
  if joint_vel.shape != joint_pos.shape:
    raise ValueError(f"joint_vel shape {joint_vel.shape} does not match joint_pos {joint_pos.shape}")
  if source_joint_names is None or len(source_joint_names) == 0:
    raise ValueError("joint_names metadata is required; shape-only motion data is not accepted")

  source = tuple(str(name) for name in source_joint_names)
  canonical = tuple(str(name) for name in canonical_joint_names)
  if len(set(source)) != len(source):
    duplicates = sorted({name for name in source if source.count(name) > 1})
    raise ValueError(f"duplicate source joint names: {duplicates}")
  if joint_pos.shape[1] != len(source):
    raise ValueError(
      f"joint array width {joint_pos.shape[1]} does not match joint_names length {len(source)}"
    )

  source_index = {name: idx for idx, name in enumerate(source)}
  missing = [name for name in canonical if name not in source_index]
  if missing:
    raise ValueError(f"missing canonical G1 23DoF joints: {missing}")

  extras = [name for name in source if name not in canonical]
  unsupported_extras = [name for name in extras if name not in _KNOWN_29DOF_EXTRAS]
  if unsupported_extras:
    raise ValueError(f"unsupported extra joints in source motion: {unsupported_extras}")

  order = [source_index[name] for name in canonical]
  projected_pos = joint_pos[:, order].astype(np.float32, copy=True)
  projected_vel = joint_vel[:, order].astype(np.float32, copy=True)
  projection = {
    "type": "name_based_23dof",
    "dropped_extra_joints": extras,
    "source_joint_count": len(source),
    "canonical_joint_count": len(canonical),
  }
  return ProjectionResult(projected_pos, projected_vel, source, canonical, projection)


def amp_obs_from_motion_arrays(
  root_pos_w: np.ndarray,
  root_quat_w: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
) -> np.ndarray:
  """Build the expert AMP observation used by the first-pass discriminator.

  The env-side AMP observation mirrors this simple contract:
  root position, root quaternion, canonical joint position, canonical joint velocity.
  """
  root_pos_w = np.asarray(root_pos_w, dtype=np.float32)
  root_quat_w = np.asarray(root_quat_w, dtype=np.float32)
  joint_pos = np.asarray(joint_pos, dtype=np.float32)
  joint_vel = np.asarray(joint_vel, dtype=np.float32)
  if root_pos_w.ndim != 2 or root_pos_w.shape[1] != 3:
    raise ValueError(f"root_pos_w must be [T, 3], got {root_pos_w.shape}")
  if root_quat_w.ndim != 2 or root_quat_w.shape[1] != 4:
    raise ValueError(f"root_quat_w must be [T, 4], got {root_quat_w.shape}")
  lengths = {root_pos_w.shape[0], root_quat_w.shape[0], joint_pos.shape[0], joint_vel.shape[0]}
  if len(lengths) != 1:
    raise ValueError(f"motion arrays have inconsistent frame counts: {sorted(lengths)}")
  return np.concatenate([root_pos_w, root_quat_w, joint_pos, joint_vel], axis=1).astype(np.float32)


def _metadata_from_npz(payload: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
  tags = _to_str_tuple(payload["tags"]) if "tags" in payload else (path.stem,)
  source = _to_str(payload["source"]) if "source" in payload else "local-fixture"
  license_name = _to_str(payload["license"]) if "license" in payload else "fixture-only"
  fps = float(np.asarray(payload["fps"]).reshape(-1)[0]) if "fps" in payload else 30.0
  return {"tags": tags, "source": source, "license": license_name, "fps": fps}


def _has_getup_tag(tags: Iterable[str], path: Path) -> bool:
  haystack = " ".join([path.stem, *[str(tag) for tag in tags]])
  return bool(_GETUP_TAG_PATTERN.search(haystack))


def _root_height_is_plausible(root_pos: np.ndarray) -> bool:
  if root_pos.shape[0] == 0:
    return False
  z = root_pos[:, 2]
  return bool(np.nanmin(z) > -0.25 and np.nanmax(z) < 2.2)


def validate_and_standardize_sequence(
  path: Path,
  output_dir: Path,
  *,
  copy_standardized: bool = True,
) -> SequenceValidationResult:
  path = Path(path)
  output_dir = Path(output_dir)
  try:
    payload = np.load(path, allow_pickle=False)
  except Exception as exc:  # pragma: no cover - exact numpy error is not stable
    return SequenceValidationResult(str(path), False, f"failed to load npz: {exc}")

  try:
    required = {"joint_pos", "root_pos_w", "root_quat_w", "joint_names"}
    missing = sorted(required - set(payload.files))
    if missing:
      raise ValueError(f"missing required fields: {missing}")
    meta = _metadata_from_npz(payload, path)
    if not _has_getup_tag(meta["tags"], path):
      raise ValueError("sequence tags/name do not indicate get-up/fall-recovery content")
    fps = float(meta["fps"])
    if not 10.0 <= fps <= 240.0:
      raise ValueError(f"unreasonable fps: {fps}")

    projection = project_to_canonical_23dof(
      payload["joint_pos"],
      payload["joint_vel"] if "joint_vel" in payload else None,
      _to_str_tuple(payload["joint_names"]),
    )
    root_pos = np.asarray(payload["root_pos_w"], dtype=np.float32)
    root_quat = np.asarray(payload["root_quat_w"], dtype=np.float32)
    amp_obs = amp_obs_from_motion_arrays(root_pos, root_quat, projection.joint_pos, projection.joint_vel)
    for name, array in {
      "joint_pos": projection.joint_pos,
      "joint_vel": projection.joint_vel,
      "root_pos_w": root_pos,
      "root_quat_w": root_quat,
      "amp_obs": amp_obs,
    }.items():
      if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if projection.joint_pos.shape[0] < 2:
      raise ValueError("sequence must contain at least two frames")
    if not _root_height_is_plausible(root_pos):
      raise ValueError("root_pos_w height is implausible for a G1 get-up clip")

    output_path: Path | None = None
    if copy_standardized:
      output_dir.mkdir(parents=True, exist_ok=True)
      output_path = output_dir / f"{path.stem}.npz"
      np.savez(
        output_path,
        joint_pos=projection.joint_pos,
        joint_vel=projection.joint_vel,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        amp_obs=amp_obs,
        joint_names=np.array(projection.canonical_joint_names),
        source_joint_names=np.array(projection.source_joint_names),
        fps=np.array([fps], dtype=np.float32),
        tags=np.array(meta["tags"]),
        source=np.array(meta["source"]),
        license=np.array(meta["license"]),
        projection=json.dumps(projection.projection, sort_keys=True),
      )

    metadata = {
      "frames": int(projection.joint_pos.shape[0]),
      "fps": fps,
      "source": meta["source"],
      "license": meta["license"],
      "tags": list(meta["tags"]),
      "joint_names": list(projection.source_joint_names),
      "canonical_joint_names": list(projection.canonical_joint_names),
      "projection": projection.projection,
      "sha256": file_sha256(path),
    }
    if output_path is not None:
      metadata["standardized_sha256"] = file_sha256(output_path)
    return SequenceValidationResult(
      str(path), True, "accepted", str(output_path) if output_path is not None else None, metadata
    )
  except Exception as exc:
    return SequenceValidationResult(str(path), False, str(exc))


def _stop_reasons_for_source_gate(
  accepted: list[SequenceValidationResult],
  *,
  source_url: str | None,
  source_revision: str | None,
  source_license: str | None,
  upstream_license: str | None,
) -> list[str]:
  reasons: list[str] = []
  if not accepted:
    reasons.append("no accepted get-up/fall-recovery candidate clips")
    return reasons

  fixture_only = all((r.metadata or {}).get("license") == "fixture-only" for r in accepted)
  if not fixture_only:
    if not source_url:
      reasons.append("source URL unresolved")
    if not source_revision:
      reasons.append("source revision unresolved")
    if not source_license:
      reasons.append("dataset-host license unresolved")
    if not upstream_license:
      reasons.append("upstream source/license restrictions unresolved")
  return reasons


def prepare_amp_dataset(
  input_dir: Path,
  output_dir: Path,
  *,
  validate_only: bool = False,
  source_url: str | None = None,
  source_revision: str | None = None,
  source_license: str | None = None,
  upstream_license: str | None = None,
) -> dict[str, Any]:
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if not input_dir.exists():
    raise FileNotFoundError(f"AMP input path does not exist: {input_dir}")
  sequence_paths = sorted(input_dir.rglob("*.npz")) if input_dir.is_dir() else [input_dir]
  results = [
    validate_and_standardize_sequence(path, output_dir / "motions", copy_standardized=True)
    for path in sequence_paths
  ]
  accepted = [r for r in results if r.accepted]
  rejected = [r for r in results if not r.accepted]
  stop_reasons = _stop_reasons_for_source_gate(
    accepted,
    source_url=source_url,
    source_revision=source_revision,
    source_license=source_license,
    upstream_license=upstream_license,
  )
  source_gate_status = "STOP" if stop_reasons else "GO"

  manifest = {
    "schema_version": _SCHEMA_VERSION,
    "input": str(input_dir),
    "validate_only": bool(validate_only),
    "canonical_joint_names": list(CANONICAL_G1_23DOF_JOINT_NAMES),
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "accepted": [asdict(r) for r in accepted],
    "rejected": [asdict(r) for r in rejected],
  }
  source_gate = {
    "schema_version": _SCHEMA_VERSION,
    "status": source_gate_status,
    "stop_reasons": stop_reasons,
    "source_url": source_url,
    "source_revision": source_revision,
    "dataset_host_license": source_license,
    "upstream_license_restrictions": upstream_license,
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
  }
  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
  (output_dir / "source_gate.json").write_text(json.dumps(source_gate, indent=2, sort_keys=True))
  return {"manifest": manifest, "source_gate": source_gate}



def source_gate_path_for_manifest(manifest_path: str | Path) -> Path:
  return Path(manifest_path).expanduser().parent / "source_gate.json"


def validate_amp_source_gate(manifest_path: str | Path) -> dict[str, Any]:
  """Return sibling source_gate.json if it exists and is GO; otherwise fail closed."""
  source_gate_path = source_gate_path_for_manifest(manifest_path)
  if not source_gate_path.exists():
    raise ValueError(
      f"AMP source gate is missing: {source_gate_path}. "
      "Run scripts/prepare_g1_getup_amp_data.py and require source_gate.status == GO before training."
    )
  try:
    source_gate = json.loads(source_gate_path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"AMP source gate is not valid JSON: {source_gate_path}: {exc}") from exc
  if source_gate.get("status") != "GO":
    reasons = source_gate.get("stop_reasons") or ["source gate status is not GO"]
    raise ValueError(f"AMP source gate blocks training: {source_gate_path}: {reasons}")
  return source_gate


class AmpExpertDataset:
  """Expert transition sampler for GetUp AMP."""

  def __init__(self, manifest_path: str | Path, device: str | torch.device = "cpu"):
    manifest_path = Path(manifest_path)
    self.source_gate = validate_amp_source_gate(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    accepted = manifest.get("accepted", [])
    if not accepted:
      raise ValueError(f"AMP manifest has no accepted sequences: {manifest_path}")
    obs_arrays = []
    for item in accepted:
      output_path = item.get("output_path")
      if not output_path:
        continue
      payload = np.load(output_path, allow_pickle=False)
      if "amp_obs" not in payload:
        payload_amp = amp_obs_from_motion_arrays(
          payload["root_pos_w"], payload["root_quat_w"], payload["joint_pos"], payload["joint_vel"]
        )
      else:
        payload_amp = payload["amp_obs"]
      obs_arrays.append(np.asarray(payload_amp, dtype=np.float32))
    if not obs_arrays:
      raise ValueError(f"AMP manifest accepted entries have no standardized output files: {manifest_path}")
    self.manifest_path = manifest_path
    self.amp_obs = torch.tensor(np.concatenate(obs_arrays, axis=0), dtype=torch.float32, device=device)
    if self.amp_obs.shape[0] < 2:
      raise ValueError("AMP expert dataset requires at least two frames")

  @property
  def obs_dim(self) -> int:
    return int(self.amp_obs.shape[1])

  def sample_observations(self, batch_size: int) -> torch.Tensor:
    idx = torch.randint(0, self.amp_obs.shape[0], (batch_size,), device=self.amp_obs.device)
    return self.amp_obs[idx]

  def sample_transitions(self, batch_size: int) -> torch.Tensor:
    max_start = self.amp_obs.shape[0] - 1
    idx = torch.randint(0, max_start, (batch_size,), device=self.amp_obs.device)
    return torch.cat([self.amp_obs[idx], self.amp_obs[idx + 1]], dim=-1)
