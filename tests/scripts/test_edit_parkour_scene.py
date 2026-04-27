from __future__ import annotations

from pathlib import Path

import pytest

from src.parkour.scene_editor import ParkourSceneDocument


SAMPLE_SCENE = """<mujoco>
  <worldbody>
    <body name="parkour_complex_terrain_course" pos="0 0 0">
      <geom name="box_a" type="box" pos="1 0 0.05" size="0.2 0.3 0.05" rgba="0.1 0.2 0.3 1" />
      <geom name="box_b" type="box" pos="2 0 0.10" size="0.2 0.3 0.10" rgba="0.4 0.5 0.6 1" />
    </body>
  </worldbody>
</mujoco>
"""


def _write_sample(path: Path) -> Path:
  path.write_text(SAMPLE_SCENE)
  return path


def test_scene_document_updates_full_dimensions_and_keeps_bottom_fixed(tmp_path: Path) -> None:
  scene_path = _write_sample(tmp_path / "scene.xml")
  document = ParkourSceneDocument.from_path(scene_path)

  document.update_module(
    "box_a",
    full_dimensions=(0.36, 1.44, 0.20),
    keep_bottom=True,
  )

  module = document.get_module("box_a")
  assert module.size == pytest.approx((0.18, 0.72, 0.10))
  assert module.pos == pytest.approx((1.0, 0.0, 0.10))


def test_scene_document_adds_deletes_and_saves_modules(tmp_path: Path) -> None:
  scene_path = _write_sample(tmp_path / "scene.xml")
  document = ParkourSceneDocument.from_path(scene_path)

  document.add_module(
    "new_block",
    pos=(3.0, 0.1, 0.0),
    full_dimensions=(0.40, 0.80, 0.12),
    rgba=(0.7, 0.6, 0.5, 1.0),
    keep_bottom=True,
  )
  document.delete_module("box_b")
  document.save()

  reloaded = ParkourSceneDocument.from_path(scene_path)
  assert reloaded.module_names() == ("box_a", "new_block")
  new_block = reloaded.get_module("new_block")
  assert new_block.pos == pytest.approx((3.0, 0.1, 0.06))
  assert new_block.size == pytest.approx((0.20, 0.40, 0.06))
  assert new_block.rgba == pytest.approx((0.7, 0.6, 0.5, 1.0))


def test_scene_document_rejects_duplicate_or_invalid_modules(tmp_path: Path) -> None:
  scene_path = _write_sample(tmp_path / "scene.xml")
  document = ParkourSceneDocument.from_path(scene_path)

  with pytest.raises(ValueError, match="already exists"):
    document.add_module("box_a")

  with pytest.raises(ValueError, match="positive"):
    document.update_module("box_a", full_dimensions=(0.0, 1.0, 1.0))
