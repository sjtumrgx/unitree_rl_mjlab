import pytest

from scripts.play_getup import build_forwarded_args


def test_play_getup_forwards_single_getup_task_with_terrain() -> None:
  assert build_forwarded_args("slope", ["--", "--checkpoint-file", "model.pt"]) == [
    "Unitree-G1-GetUp",
    "--getup-terrain=slope",
    "--checkpoint-file",
    "model.pt",
  ]


def test_play_getup_rejects_prone() -> None:
  with pytest.raises(ValueError):
    build_forwarded_args("ground_prone", [])
