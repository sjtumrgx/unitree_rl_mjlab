"""Inventory checks for removing the old GetUp / AntiFall-GetUp lanes."""

from pathlib import Path

REMOVED_PATHS = (
  "data/g1_getup_amp.yaml",
  "deploy/robots/g1_getup",
  "doc/g1_antifall_getup.md",
  "doc/g1_getup.md",
  "doc/g1_getup_demo_data.md",
  "scripts/diagnose_antifall_getup_rollout.py",
  "scripts/diagnose_getup_contracts.py",
  "scripts/diagnose_getup_multiterrain.py",
  "scripts/diagnose_getup_rollout.py",
  "scripts/evaluate_getup_amp.py",
  "scripts/evaluate_getup_amp_style.py",
  "scripts/g1_getup_amp_config.py",
  "scripts/play_g1_getup_amp_data.py",
  "scripts/play_getup.py",
  "scripts/prepare_g1_getup_amp_data.py",
  "scripts/train_getup.py",
  "scripts/train_getup_amp.py",
  "src/assets/motions/g1/getup_synthetic.csv",
  "src/assets/motions/g1/getup_synthetic_demo.npz",
  "src/tasks/velocity/config/g1_getup",
  "src/tasks/velocity/mdp/getup",
  "src/tasks/velocity/rl/getup_amp.py",
  "src/tasks/velocity/rl/getup_amp_data.py",
  "tests/fixtures/g1_getup_amp",
  "tests/scripts/test_diagnose_antifall_getup_rollout.py",
  "tests/scripts/test_diagnose_getup_contracts.py",
  "tests/scripts/test_diagnose_getup_multiterrain.py",
  "tests/scripts/test_diagnose_getup_rollout.py",
  "tests/scripts/test_evaluate_getup_amp_style.py",
  "tests/scripts/test_play_g1_getup_amp_data.py",
  "tests/scripts/test_prepare_g1_getup_amp_data.py",
  "tests/scripts/test_train_getup.py",
  "tests/scripts/test_train_getup_amp.py",
  "tests/tasks/test_g1_antifall_getup_contract.py",
  "tests/tasks/test_g1_antifall_getup_deploy_contract.py",
  "tests/tasks/test_g1_getup_amp_algorithm.py",
  "tests/tasks/test_g1_getup_amp_contract.py",
  "tests/tasks/test_g1_getup_nan_safety.py",
  "tests/tasks/test_g1_getup_reset_and_assist_contracts.py",
)

REMOVED_TEXT = (
  "Unitree-G1-GetUp",
  "Unitree-G1-AntiFall-GetUp",
  "train_getup.py",
  "play_getup.py",
  "g1_getup",
  "AntiFall-GetUp",
)


def test_removed_getup_lane_files_are_absent() -> None:
  leftovers = [path for path in REMOVED_PATHS if Path(path).exists()]
  assert leftovers == []


def test_readmes_no_longer_advertise_removed_getup_lanes() -> None:
  for path in (Path("README.md"), Path("README_zh.md"), *Path("doc").glob("*.md")):
    text = path.read_text(encoding="utf-8")
    leftovers = [needle for needle in REMOVED_TEXT if needle in text]
    assert leftovers == [], f"{path} still contains {leftovers}"
