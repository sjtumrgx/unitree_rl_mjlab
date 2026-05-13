"""Data contracts for optional G1 GetUp AMP demonstrations.

The AMP path intentionally keeps third-party motion data outside git.  This module
normalizes small local motion files into a manifest-backed schema that the AMP
algorithm can consume deterministically in tests and in offline training jobs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
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

_OPENHE_GETUP_NAME_PATTERN = re.compile(
  r"(get.?up|lie.?to.?crouch|crouch.?to.?walk|crouch.?to.?run|stand.?up|recovery)",
  re.I,
)


def canonical_g1_23dof_joint_names() -> tuple[str, ...]:
  """Return the active G1 23DoF joint order from the checked-in MJCF."""
  root = ET.parse(_G1_23DOF_XML).getroot()
  return tuple(
    joint.attrib["name"]
    for joint in root.findall(".//joint")
    if "name" in joint.attrib and joint.attrib["name"] != "floating_base_joint"
  )


CANONICAL_G1_23DOF_JOINT_NAMES = canonical_g1_23dof_joint_names()
AMP_OBS_DIM = 1 + 4 + 2 * len(CANONICAL_G1_23DOF_JOINT_NAMES)


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


def _yaw_invariant_quat_wxyz_np(quat_wxyz: np.ndarray) -> np.ndarray:
  """Numpy equivalent of the env-side yaw-invariant quaternion transform."""
  q = np.asarray(quat_wxyz, dtype=np.float32)
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  half = -0.5 * yaw
  yw, yz = np.cos(half), np.sin(half)
  new_w = yw * w - yz * z
  new_x = yw * x - yz * y
  new_y = yw * y + yz * x
  new_z = yw * z + yz * w
  out = np.stack([new_w, new_x, new_y, new_z], axis=-1).astype(np.float32)
  norm = np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-6, None)
  return out / norm


def amp_obs_yaw_invariant(
  root_pos_w: np.ndarray,
  root_quat_w: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
) -> np.ndarray:
  """Heading-invariant AMP features: [root_z, yaw_free_quat, joint_pos, joint_vel].

  Mirrors `src/tasks/velocity/mdp/getup/amp_observations.py:amp_getup_features`
  so demo and env observations live in the same distribution.  Dropping XY
  removes a confound that previously let the discriminator separate by
  absolute world position alone.
  """
  root_pos_w = np.asarray(root_pos_w, dtype=np.float32)
  root_quat_w = np.asarray(root_quat_w, dtype=np.float32)
  joint_pos = np.asarray(joint_pos, dtype=np.float32)
  joint_vel = np.asarray(joint_vel, dtype=np.float32)
  z = root_pos_w[:, 2:3]
  quat = _yaw_invariant_quat_wxyz_np(root_quat_w)
  return np.concatenate([z, quat, joint_pos, joint_vel], axis=1).astype(np.float32)


def _resample_motion_to_dt(
  joint_pos: np.ndarray,
  root_pos_w: np.ndarray,
  root_quat_w: np.ndarray,
  source_fps: float,
  target_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Resample a clip from source_fps to 1/target_dt and recompute joint_vel.

  Linear interpolation is sufficient because the resulting joint_vel is
  computed by finite differences over the resampled grid, matching the env's
  per-step velocity time scale.
  """
  source_dt = 1.0 / float(source_fps)
  num_source = int(joint_pos.shape[0])
  duration = (num_source - 1) * source_dt
  num_target = max(2, int(round(duration / float(target_dt))) + 1)
  t_source = np.arange(num_source, dtype=np.float32) * source_dt
  t_target = np.linspace(0.0, duration, num_target, dtype=np.float32)

  def _interp(arr: np.ndarray) -> np.ndarray:
    out = np.empty((num_target, arr.shape[1]), dtype=np.float32)
    for c in range(arr.shape[1]):
      out[:, c] = np.interp(t_target, t_source, arr[:, c])
    return out

  jp = _interp(joint_pos)
  rp = _interp(root_pos_w)
  rq = _interp(root_quat_w)
  rq = rq / np.clip(np.linalg.norm(rq, axis=1, keepdims=True), 1e-6, None)
  jv = np.gradient(jp, axis=0).astype(np.float32) / float(target_dt)
  return jp, jv, rp, rq.astype(np.float32)


def _extract_getup_segments(
  root_pos_w: np.ndarray,
  *,
  fallen_height: float = 0.30,
  standing_height: float = 0.55,
  pad_frames: int = 5,
) -> list[tuple[int, int]]:
  """Find contiguous frame ranges spanning an actual fall->stand-up rise.

  Random sampling over the full LAFAN clip is dominated by the standing-walking
  segments; the discriminator learns 'be tall' rather than 'perform a get-up'.
  Returning explicit (start, end) ranges that contain a transition from
  ``z < fallen_height`` up through ``z >= standing_height`` lets the dataset
  sample inside each segment without straddling unrelated content.
  """
  z = np.asarray(root_pos_w, dtype=np.float32)[:, 2]
  n = z.shape[0]
  segments: list[tuple[int, int]] = []
  i = 0
  while i < n:
    if z[i] < fallen_height:
      j = i
      while j < n and z[j] < standing_height:
        j += 1
      if j < n:  # only keep segments that actually reach standing
        start = max(0, i - pad_frames)
        end = min(n - 1, j + pad_frames)
        segments.append((start, end))
        i = end + 1
      else:
        break
    else:
      i += 1
  return segments


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



def _write_standardized_npz(
  output_path: Path,
  *,
  projection: ProjectionResult,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  amp_obs: np.ndarray,
  fps: float,
  tags: Sequence[str],
  source: str,
  license_name: str,
) -> None:
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
    tags=np.array(tuple(tags)),
    source=np.array(source),
    license=np.array(license_name),
    projection=json.dumps(projection.projection, sort_keys=True),
  )


def _standardized_metadata(
  *,
  path: Path,
  projection: ProjectionResult,
  output_path: Path | None,
  frame_count: int,
  fps: float,
  tags: Sequence[str],
  source: str,
  license_name: str,
) -> dict[str, Any]:
  metadata = {
    "frames": int(frame_count),
    "fps": float(fps),
    "source": source,
    "license": license_name,
    "tags": list(tags),
    "joint_names": list(projection.source_joint_names),
    "canonical_joint_names": list(projection.canonical_joint_names),
    "projection": projection.projection,
    "sha256": file_sha256(path),
  }
  if output_path is not None:
    metadata["standardized_sha256"] = file_sha256(output_path)
  return metadata


def _validate_common_arrays(
  *,
  projection: ProjectionResult,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  amp_obs: np.ndarray,
) -> None:
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


def _validate_and_standardize_openhe_pkl(
  path: Path,
  output_dir: Path,
  *,
  copy_standardized: bool = True,
) -> SequenceValidationResult:
  try:
    payload = _load_openhe_pickle_payload(path)
  except Exception as exc:  # pragma: no cover - pickle errors are data-dependent
    return SequenceValidationResult(str(path), False, f"failed to load pkl: {exc}")
  try:
    if not isinstance(payload, dict) or not payload:
      raise ValueError("OpenHE pkl must contain a non-empty motion dictionary")
    motion_key = next(iter(payload))
    motion = payload[motion_key]
    required = {"root_trans_offset", "root_rot", "dof", "fps"}
    missing = sorted(required - set(motion))
    if missing:
      raise ValueError(f"missing required OpenHE fields: {missing}")
    tags = (path.stem, str(motion_key))
    if not _OPENHE_GETUP_NAME_PATTERN.search(" ".join(tags)):
      raise ValueError("OpenHE clip name does not indicate get-up/fall-recovery content")
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    if not 10.0 <= fps <= 240.0:
      raise ValueError(f"unreasonable fps: {fps}")
    joint_pos = np.asarray(motion["dof"], dtype=np.float32)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(CANONICAL_G1_23DOF_JOINT_NAMES):
      raise ValueError(f"OpenHE dof must be [T, 23], got {joint_pos.shape}")
    joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32) * fps
    projection = project_to_canonical_23dof(
      joint_pos,
      joint_vel,
      CANONICAL_G1_23DOF_JOINT_NAMES,
    )
    projection.projection["source_format"] = "openhe_g1_retargeted_pkl"
    projection.projection["joint_order_assumption"] = (
      "OpenHE Unitree G1 23DoF README dof order matches active g1_23dof.xml"
    )
    root_pos = np.asarray(motion["root_trans_offset"], dtype=np.float32)
    root_quat = _openhe_quat_xyzw_to_wxyz(
      np.asarray(motion["root_rot"], dtype=np.float32)
    )
    amp_obs = amp_obs_from_motion_arrays(root_pos, root_quat, projection.joint_pos, projection.joint_vel)
    _validate_common_arrays(projection=projection, root_pos=root_pos, root_quat=root_quat, amp_obs=amp_obs)

    output_path: Path | None = None
    if copy_standardized:
      output_dir.mkdir(parents=True, exist_ok=True)
      output_path = output_dir / f"{path.stem}.npz"
      _write_standardized_npz(
        output_path,
        projection=projection,
        root_pos=root_pos,
        root_quat=root_quat,
        amp_obs=amp_obs,
        fps=fps,
        tags=tags,
        source="openhe/g1-retargeted-motions",
        license_name="MIT + original source restrictions",
      )
    metadata = _standardized_metadata(
      path=path,
      projection=projection,
      output_path=output_path,
      frame_count=projection.joint_pos.shape[0],
      fps=fps,
      tags=tags,
      source="openhe/g1-retargeted-motions",
      license_name="MIT + original source restrictions",
    )
    return SequenceValidationResult(
      str(path), True, "accepted", str(output_path) if output_path is not None else None, metadata
    )
  except Exception as exc:
    return SequenceValidationResult(str(path), False, str(exc))


def _load_openhe_pickle_payload(path: Path) -> Any:
  """Load OpenHE `.pkl` files produced by joblib while keeping fixture support.

  OpenHE stores large NumPy arrays with joblib's `NumpyArrayWrapper`, which a
  plain `pickle.load` cannot fully consume.  Synthetic tests and hand-authored
  small clips may still be regular pickle files, so keep pickle as fallback.
  """
  try:
    import joblib
  except ModuleNotFoundError:
    with path.open("rb") as f:
      try:
        return pickle.load(f)
      except Exception as pickle_exc:  # pragma: no cover - depends on external data encoding
        raise ModuleNotFoundError(
          "OpenHE retargeted `.pkl` files require the optional `joblib` package; "
          "install this project dependency or run `python -m pip install joblib`."
        ) from pickle_exc

  try:
    return joblib.load(path)
  except Exception:
    with path.open("rb") as f:
      return pickle.load(f)


def _openhe_quat_xyzw_to_wxyz(root_quat_xyzw: np.ndarray) -> np.ndarray:
  root_quat_xyzw = np.asarray(root_quat_xyzw, dtype=np.float32)
  if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
    raise ValueError(
      f"OpenHE root_rot must be [T, 4] xyzw, got {root_quat_xyzw.shape}"
    )
  root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32, copy=True)
  norms = np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)
  if np.any(norms <= 1e-8):
    raise ValueError("OpenHE root_rot contains a near-zero quaternion")
  return root_quat_wxyz / norms

def validate_and_standardize_sequence(
  path: Path,
  output_dir: Path,
  *,
  copy_standardized: bool = True,
) -> SequenceValidationResult:
  path = Path(path)
  output_dir = Path(output_dir)
  if path.suffix.lower() == ".pkl":
    return _validate_and_standardize_openhe_pkl(
      path, output_dir, copy_standardized=copy_standardized
    )
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
    _validate_common_arrays(projection=projection, root_pos=root_pos, root_quat=root_quat, amp_obs=amp_obs)

    output_path: Path | None = None
    if copy_standardized:
      output_dir.mkdir(parents=True, exist_ok=True)
      output_path = output_dir / f"{path.stem}.npz"
      _write_standardized_npz(
        output_path,
        projection=projection,
        root_pos=root_pos,
        root_quat=root_quat,
        amp_obs=amp_obs,
        fps=fps,
        tags=meta["tags"],
        source=meta["source"],
        license_name=meta["license"],
      )

    metadata = _standardized_metadata(
      path=path,
      projection=projection,
      output_path=output_path,
      frame_count=projection.joint_pos.shape[0],
      fps=fps,
      tags=meta["tags"],
      source=meta["source"],
      license_name=meta["license"],
    )
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
  if not input_dir.exists():
    raise FileNotFoundError(f"AMP input path does not exist: {input_dir}")
  sequence_paths = (
    sorted([*input_dir.rglob("*.npz"), *input_dir.rglob("*.pkl")])
    if input_dir.is_dir()
    else [input_dir]
  )
  return prepare_amp_sequences(
    sequence_paths,
    output_dir,
    input_label=str(input_dir),
    validate_only=validate_only,
    source_url=source_url,
    source_revision=source_revision,
    source_license=source_license,
    upstream_license=upstream_license,
  )


def prepare_amp_sequences(
  sequence_paths: Sequence[Path],
  output_dir: Path,
  *,
  input_label: str | None = None,
  validate_only: bool = False,
  source_url: str | None = None,
  source_revision: str | None = None,
  source_license: str | None = None,
  upstream_license: str | None = None,
) -> dict[str, Any]:
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  sequence_paths = [Path(path) for path in sequence_paths]
  missing = [str(path) for path in sequence_paths if not path.exists()]
  if missing:
    raise FileNotFoundError(f"AMP input sequence path(s) do not exist: {missing}")
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
    "input": (
      input_label if input_label is not None else [str(path) for path in sequence_paths]
    ),
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
  """Expert transition sampler for GetUp AMP.

  Three correctness invariants over the legacy implementation:
    1. Per-sequence boundary tracking — sampled (t, t+1) pairs never cross
       motion files concatenated end-to-end.
    2. Resample each motion to ``target_dt`` so the demo and env share the
       same temporal scale.  Joint velocity is recomputed by finite differences
       at the new spacing instead of trusting the source's ``np.gradient * fps``.
    3. Use a yaw-invariant feature layout (drops world XY, removes yaw from
       root_quat) so policy and demo observations live in the same heading
       frame.  See ``amp_obs_yaw_invariant``.

  When ``getup_segments=True`` only frame ranges that actually contain a fall
  -> stand-up transition are kept, so random pair samples reflect the
  recovery skill rather than the dominant standing-walking content.
  """

  def __init__(
    self,
    manifest_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    target_dt: float = 0.02,
    getup_segments: bool = True,
    fallen_height: float = 0.30,
    standing_height: float = 0.55,
    feature_layout: str = "yaw_invariant",
  ):
    manifest_path = Path(manifest_path)
    self.source_gate = validate_amp_source_gate(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    accepted = manifest.get("accepted", [])
    if not accepted:
      raise ValueError(f"AMP manifest has no accepted sequences: {manifest_path}")

    obs_arrays: list[np.ndarray] = []
    boundaries: list[int] = []
    cursor = 0
    for item in accepted:
      output_path = item.get("output_path")
      if not output_path:
        continue
      payload = np.load(output_path, allow_pickle=False)
      joint_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
      root_pos = np.asarray(payload["root_pos_w"], dtype=np.float32)
      root_quat = np.asarray(payload["root_quat_w"], dtype=np.float32)
      source_fps = float(np.asarray(payload["fps"]).reshape(-1)[0])

      # Resample to env dt; recompute joint_vel at target spacing.
      joint_pos_r, joint_vel_r, root_pos_r, root_quat_r = _resample_motion_to_dt(
        joint_pos, root_pos, root_quat, source_fps=source_fps, target_dt=float(target_dt)
      )

      # Optionally crop to actual fall->stand segments.
      if getup_segments:
        segments = _extract_getup_segments(
          root_pos_r, fallen_height=fallen_height, standing_height=standing_height
        )
      else:
        segments = [(0, joint_pos_r.shape[0] - 1)]

      for start, end in segments:
        if end - start < 2:
          continue
        if feature_layout == "yaw_invariant":
          seg_obs = amp_obs_yaw_invariant(
            root_pos_r[start : end + 1],
            root_quat_r[start : end + 1],
            joint_pos_r[start : end + 1],
            joint_vel_r[start : end + 1],
          )
        else:
          seg_obs = amp_obs_from_motion_arrays(
            root_pos_r[start : end + 1],
            root_quat_r[start : end + 1],
            joint_pos_r[start : end + 1],
            joint_vel_r[start : end + 1],
          )
        obs_arrays.append(np.asarray(seg_obs, dtype=np.float32))
        cursor += seg_obs.shape[0]
        boundaries.append(cursor)

    if not obs_arrays:
      raise ValueError(
        f"AMP manifest produced no usable get-up segments: {manifest_path}. "
        "Check fallen/standing height thresholds or disable getup_segments."
      )
    self.manifest_path = manifest_path
    self.target_dt = float(target_dt)
    self.feature_layout = str(feature_layout)
    self.amp_obs = torch.tensor(np.concatenate(obs_arrays, axis=0), dtype=torch.float32, device=device)
    if self.amp_obs.shape[0] < 2:
      raise ValueError("AMP expert dataset requires at least two frames")

    # boundaries[k] = exclusive end index of sequence k; previous = inclusive start.
    starts = [0, *boundaries[:-1]]
    valid_pair_starts: list[int] = []
    for s, e in zip(starts, boundaries):
      # last frame of each sequence has no valid t+1 partner -> drop it.
      valid_pair_starts.extend(range(s, max(s, e - 1)))
    if not valid_pair_starts:
      raise ValueError("AMP expert dataset has no within-sequence transitions")
    self._pair_start_indices = torch.tensor(valid_pair_starts, dtype=torch.long, device=device)

  @property
  def obs_dim(self) -> int:
    return int(self.amp_obs.shape[1])

  def sample_observations(self, batch_size: int) -> torch.Tensor:
    idx = torch.randint(0, self.amp_obs.shape[0], (batch_size,), device=self.amp_obs.device)
    return self.amp_obs[idx]

  def sample_transitions(self, batch_size: int) -> torch.Tensor:
    pool = self._pair_start_indices
    pick = torch.randint(0, pool.shape[0], (int(batch_size),), device=pool.device)
    starts = pool[pick]
    return torch.cat([self.amp_obs[starts], self.amp_obs[starts + 1]], dim=-1)
