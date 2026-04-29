"""Train the opt-in ground-only Unitree-G1-GetUp-AMP fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

AMP_TASK_ID = "Unitree-G1-GetUp-AMP"

from src.tasks.velocity.rl.getup_amp_data import validate_amp_source_gate


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--demo-data-dir", default="data/motions/g1_getup_amp")
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
  validate_demo_data_dir(args.demo_data_dir, args.manifest_path)
  from scripts.train import main as train_main

  sys.argv = [
    "scripts/train.py",
    *build_forwarded_args(
      demo_data_dir=args.demo_data_dir,
      manifest_path=args.manifest_path,
      max_iterations=args.max_iterations,
      num_envs=args.num_envs,
      headless_smoke=args.headless_smoke,
      extra_args=args.extra_args,
    ),
  ]
  train_main()


if __name__ == "__main__":
  main()
