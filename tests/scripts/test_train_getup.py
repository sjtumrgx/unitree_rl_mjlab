import pytest

from scripts.train_getup import build_forwarded_args


def test_train_getup_forwards_single_getup_task_with_terrain() -> None:
  assert build_forwarded_args("platform", ["--", "--agent.max-iterations=1"]) == [
    "Unitree-G1-GetUp",
    "--getup-terrain=platform",
    "--agent.max-iterations=1",
  ]


def test_train_getup_rejects_prone() -> None:
  with pytest.raises(ValueError):
    build_forwarded_args("ground_prone", [])
