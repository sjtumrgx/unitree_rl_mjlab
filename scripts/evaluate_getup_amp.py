"""Write a short headless diagnostic JSON for the G1 GetUp AMP data path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from src.tasks.velocity.rl.getup_amp import AmpDiscriminator
from src.tasks.velocity.rl.getup_amp_data import AmpExpertDataset


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--demo-data-dir", default="data/motions/g1_getup_amp")
  parser.add_argument("--manifest-path", default=None)
  parser.add_argument("--policy-mode", choices=("random",), default="random")
  parser.add_argument("--compare-no-demo", action="store_true")
  parser.add_argument("--max-steps", type=int, default=200)
  parser.add_argument("--viewer", choices=("none", "native", "viser"), default="none")
  parser.add_argument("--output", required=True, type=Path)
  return parser


def _manifest_path(demo_data_dir: str, manifest_path: str | None) -> Path:
  return Path(manifest_path).expanduser() if manifest_path else Path(demo_data_dir).expanduser() / "manifest.json"


def _first_motion_path(manifest: dict) -> Path:
  accepted = manifest.get("accepted", [])
  if not accepted:
    raise ValueError("AMP manifest has no accepted clips")
  path = accepted[0].get("output_path")
  if not path:
    raise ValueError("AMP accepted clip has no standardized output_path")
  return Path(path)


def build_diagnostic(args: argparse.Namespace) -> dict:
  manifest_path = _manifest_path(args.demo_data_dir, args.manifest_path)
  manifest = json.loads(manifest_path.read_text())
  source_gate_path = manifest_path.parent / "source_gate.json"
  source_gate = json.loads(source_gate_path.read_text()) if source_gate_path.exists() else {"status": "UNKNOWN"}
  motion_path = _first_motion_path(manifest)
  motion = np.load(motion_path, allow_pickle=False)
  max_steps = min(int(args.max_steps), int(motion["amp_obs"].shape[0]))
  amp_obs = torch.tensor(motion["amp_obs"][:max_steps], dtype=torch.float32)
  transitions = torch.cat([amp_obs[:-1], amp_obs[1:]], dim=-1) if max_steps > 1 else torch.cat([amp_obs, amp_obs], dim=-1)
  torch.manual_seed(0)
  discriminator = AmpDiscriminator(transitions.shape[-1], hidden_dims=(32,))
  with torch.no_grad():
    amp_score = torch.sigmoid(discriminator(transitions)).mean().item()
    amp_reward = discriminator.reward(transitions).mean().item()
  root_pos = motion["root_pos_w"][:max_steps]
  root_quat = motion["root_quat_w"][:max_steps]
  # For wxyz quaternions, upright identity-like examples have high |w|.  This
  # diagnostic is intentionally data-path level; full physics validation remains
  # in train/play smoke runs.
  upright_alignment = float(np.clip(np.abs(root_quat[:, 0]).mean(), 0.0, 1.0))
  return {
    "run_mode": "amp",
    "policy_mode": args.policy_mode,
    "compare_no_demo": bool(args.compare_no_demo),
    "viewer": args.viewer,
    "manifest_path": str(manifest_path),
    "source_gate_status": source_gate.get("status"),
    "steps": max_steps,
    "torso_height_mean": float(root_pos[:, 2].mean()),
    "torso_height_final": float(root_pos[-1, 2]),
    "upright_alignment_mean": upright_alignment,
    "feet_contact_mean": None,
    "support_contact_mean": None,
    "terminated": False,
    "termination_reason": None,
    "amp_score": float(amp_score),
    "amp_reward_mean": float(amp_reward),
  }


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  # Validate manifest with the same dataset loader used by training.
  AmpExpertDataset(_manifest_path(args.demo_data_dir, args.manifest_path))
  diagnostic = build_diagnostic(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True))
  print(json.dumps(diagnostic, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
