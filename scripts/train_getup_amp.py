"""Train the opt-in ground-only Unitree-G1-GetUp-AMP fallback."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

AMP_TASK_ID = "Unitree-G1-GetUp-AMP"

from src.tasks.velocity.rl.getup_amp_data import validate_amp_source_gate
from scripts.g1_getup_amp_config import (
  DEFAULT_CONFIG_PATH,
  load_workflow_config,
  path_list,
  repo_path,
  section,
)


@dataclass(frozen=True)
class TrainSettings:
  demo_data_dir: str
  manifest_path: str | None
  npz_files: tuple[str, ...]
  max_iterations: int | None
  num_envs: int | None


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default=DEFAULT_CONFIG_PATH,
    type=Path,
    help=(
      "YAML workflow config. Defaults to "
      f"{DEFAULT_CONFIG_PATH.as_posix()}."
    ),
  )
  parser.add_argument("--demo-data-dir", default=None)
  parser.add_argument("--manifest-path", default=None)
  parser.add_argument("--max-iterations", type=int, default=None)
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument(
    "--headless-smoke",
    action="store_true",
    help="Use CPU/tensorboard/no-upload defaults suitable for a one-iteration smoke run.",
  )
  parser.add_argument("extra_args", nargs=argparse.REMAINDER)
  return parser


def _strip_remainder_separator(extra_args: list[str]) -> list[str]:
  return extra_args[1:] if extra_args[:1] == ["--"] else extra_args


def resolve_train_settings(
  *,
  config_path: str | Path | None,
  demo_data_dir: str | None,
  manifest_path: str | None,
  max_iterations: int | None,
  num_envs: int | None,
) -> TrainSettings:
  config = load_workflow_config(config_path)
  train_cfg = section(config, "train")
  resolved_demo_data_dir = (
    demo_data_dir
    if demo_data_dir is not None
    else str(train_cfg.get("demo_data_dir", "data/motions/g1_getup_amp"))
  )
  resolved_manifest_path = (
    manifest_path
    if manifest_path is not None
    else train_cfg.get("manifest_path")
  )
  resolved_npz_files = tuple(str(repo_path(path)) for path in path_list(train_cfg.get("npz_files")))
  resolved_max_iterations = (
    max_iterations
    if max_iterations is not None
    else train_cfg.get("max_iterations")
  )
  resolved_num_envs = (
    num_envs
    if num_envs is not None
    else train_cfg.get("num_envs")
  )
  return TrainSettings(
    demo_data_dir=str(resolved_demo_data_dir),
    manifest_path=str(resolved_manifest_path)
    if resolved_manifest_path is not None
    else None,
    npz_files=resolved_npz_files,
    max_iterations=int(resolved_max_iterations)
    if resolved_max_iterations is not None
    else None,
    num_envs=int(resolved_num_envs) if resolved_num_envs is not None else None,
  )


def write_selected_train_manifest(
  *,
  demo_data_dir: str,
  npz_files: list[str] | tuple[str, ...],
  manifest_name: str = "selected_manifest.json",
) -> Path:
  """Write a manifest that trains only on the YAML-selected `.npz` files."""
  demo_dir = Path(demo_data_dir).expanduser()
  base_manifest_path = demo_dir / "manifest.json"
  if not base_manifest_path.exists():
    raise FileNotFoundError(
      "Base AMP manifest not found. Run scripts/prepare_g1_getup_amp_data.py first; "
      f"expected {base_manifest_path}"
    )
  selected_paths = [repo_path(path) for path in npz_files]
  missing = [str(path) for path in selected_paths if not path.exists()]
  if missing:
    raise FileNotFoundError(f"Configured train.npz_files do not exist: {missing}")

  base_manifest = json.loads(base_manifest_path.read_text())
  selected_by_resolved_path = {path.resolve(): path for path in selected_paths}
  selected_entries = []
  matched: set[Path] = set()
  for item in base_manifest.get("accepted", []):
    output_path = item.get("output_path")
    if not output_path:
      continue
    resolved_output = Path(output_path).expanduser().resolve()
    if resolved_output in selected_by_resolved_path:
      selected_entries.append(item)
      matched.add(resolved_output)

  for resolved_path, original_path in selected_by_resolved_path.items():
    if resolved_path in matched:
      continue
    selected_entries.append(
      {
        "path": str(original_path),
        "accepted": True,
        "reason": "selected for training",
        "output_path": str(original_path),
        "metadata": None,
      }
    )

  if not selected_entries:
    raise ValueError("train.npz_files did not select any usable AMP clips")

  selected_manifest = dict(base_manifest)
  selected_manifest["input"] = [str(path) for path in selected_paths]
  selected_manifest["accepted"] = selected_entries
  selected_manifest["accepted_count"] = len(selected_entries)
  selected_manifest["rejected"] = []
  selected_manifest["rejected_count"] = 0

  selected_manifest_path = demo_dir / manifest_name
  selected_manifest_path.write_text(json.dumps(selected_manifest, indent=2, sort_keys=True))
  return selected_manifest_path


def validate_demo_data_dir(demo_data_dir: str, manifest_path: str | None = None) -> Path:
  manifest = Path(manifest_path).expanduser() if manifest_path else Path(demo_data_dir).expanduser() / "manifest.json"
  if not manifest.exists():
    raise FileNotFoundError(
      "AMP demo manifest not found. Run scripts/play_g1_getup_amp_data.py --validate-only "
      "or scripts/prepare_g1_getup_amp_data.py first; "
      f"expected {manifest}"
    )
  validate_amp_source_gate(manifest)
  return manifest


def build_forwarded_args(
  *,
  demo_data_dir: str,
  manifest_path: str | None,
  max_iterations: int | None,
  num_envs: int | None,
  headless_smoke: bool,
  extra_args: list[str],
) -> list[str]:
  forwarded = [
    AMP_TASK_ID,
    f"--agent.algorithm.demo-data-dir={demo_data_dir}",
  ]
  if manifest_path is not None:
    forwarded.append(f"--agent.algorithm.manifest-path={manifest_path}")
  if max_iterations is not None:
    forwarded.append(f"--agent.max-iterations={max_iterations}")
  if num_envs is not None:
    forwarded.append(f"--env.scene.num-envs={num_envs}")
  if headless_smoke:
    forwarded.extend([
      "--gpu-ids=cpu",
      "--agent.logger=tensorboard",
      "--agent.upload-model=False",
      "--agent.save-interval=1000000",
    ])
  forwarded.extend(_strip_remainder_separator(extra_args))
  return forwarded


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  settings = resolve_train_settings(
    config_path=args.config,
    demo_data_dir=args.demo_data_dir,
    manifest_path=args.manifest_path,
    max_iterations=args.max_iterations,
    num_envs=args.num_envs,
  )
  manifest_path = settings.manifest_path
  if manifest_path is None and settings.npz_files:
    manifest_path = str(
      write_selected_train_manifest(
        demo_data_dir=settings.demo_data_dir,
        npz_files=settings.npz_files,
      )
    )
  validate_demo_data_dir(settings.demo_data_dir, manifest_path)
  from scripts.train import main as train_main

  sys.argv = [
    "scripts/train.py",
    *build_forwarded_args(
      demo_data_dir=settings.demo_data_dir,
      manifest_path=manifest_path,
      max_iterations=settings.max_iterations,
      num_envs=settings.num_envs,
      headless_smoke=args.headless_smoke,
      extra_args=args.extra_args,
    ),
  ]
  train_main()


if __name__ == "__main__":
  main()
