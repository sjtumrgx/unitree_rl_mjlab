from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


def _load_frames(capture_npz: Path, expected_size: int) -> np.ndarray:
  payload = np.load(capture_npz, allow_pickle=True)
  if "support_depth" not in payload:
    raise ValueError("Expected 'support_depth' array in NPZ payload.")
  frames = np.asarray(payload["support_depth"], dtype=np.float32)
  if frames.ndim == 1:
    frames = frames.reshape(1, -1)
  elif frames.ndim == 3:
    frames = frames.reshape(frames.shape[0], -1)
  elif frames.ndim != 2:
    raise ValueError("'support_depth' must be a 1D, 2D, or 3D array.")
  if frames.shape[1] != expected_size:
    raise ValueError(
      f"Expected flattened support_depth frame size {expected_size}, got {frames.shape[1]}."
    )
  return frames


def _patch_shape_from_deploy_yaml(deploy_yaml: Path) -> tuple[int, int]:
  payload = yaml.safe_load(deploy_yaml.read_text())
  patch_shape = payload["support_geometry_interface"]["patch_shape"]
  if not isinstance(patch_shape, list) or len(patch_shape) != 2:
    raise ValueError("deploy.yaml support_geometry_interface.patch_shape must be a 2-item list.")
  return int(patch_shape[0]), int(patch_shape[1])


def summarize_depth_capture(*, capture_npz: Path, deploy_yaml: Path) -> dict[str, Any]:
  patch_h, patch_w = _patch_shape_from_deploy_yaml(deploy_yaml)
  frames = _load_frames(capture_npz, patch_h * patch_w)
  return {
    "num_frames": int(frames.shape[0]),
    "patch_shape": [patch_h, patch_w],
    "frame_size": int(frames.shape[1]),
    "min_value": float(frames.min()),
    "max_value": float(frames.max()),
    "mean_value": float(frames.mean()),
    "zero_fraction": float((frames == 0.0).sum() / frames.size),
  }


def save_depth_artifacts(*, capture_npz: Path, deploy_yaml: Path, artifact_dir: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  patch_h, patch_w = _patch_shape_from_deploy_yaml(deploy_yaml)
  frames = _load_frames(capture_npz, patch_h * patch_w).reshape(-1, patch_h, patch_w)
  artifact_dir.mkdir(parents=True, exist_ok=True)

  images = {
    "first_frame.png": frames[0],
    "last_frame.png": frames[-1],
    "mean_frame.png": frames.mean(axis=0),
  }
  for filename, image in images.items():
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(image, cmap="viridis")
    ax.set_title(filename.replace("_", " ").replace(".png", ""))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(artifact_dir / filename, dpi=160)
    plt.close(fig)


def _print_json(payload: dict[str, Any], *, output: Path | None = None) -> None:
  text = json.dumps(payload, indent=2, sort_keys=True)
  if output is not None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
  print(text)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Inspect topology-getup support-depth captures and save calibration visuals.",
  )
  parser.add_argument("capture_npz", type=Path)
  parser.add_argument(
    "--deploy-yaml",
    type=Path,
    default=Path("deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml"),
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--artifact-dir", type=Path)
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  summary = summarize_depth_capture(capture_npz=args.capture_npz, deploy_yaml=args.deploy_yaml)
  if args.artifact_dir is not None:
    save_depth_artifacts(
      capture_npz=args.capture_npz,
      deploy_yaml=args.deploy_yaml,
      artifact_dir=args.artifact_dir,
    )
  _print_json(summary, output=args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
