"""Editable MJCF terrain-module helpers for the G1 parkour scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from typing import Iterable, Sequence

DEFAULT_BODY_NAME = "parkour_complex_terrain_course"
DEFAULT_SCENE_PATH = (
  Path(__file__).resolve().parents[1]
  / "assets"
  / "robots"
  / "unitree_g1"
  / "xmls"
  / "scene_g1_parkour.xml"
)


Float3 = tuple[float, float, float]
Float4 = tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneModule:
  """One editable box geom in the parkour MJCF terrain body."""

  name: str
  pos: Float3
  size: Float3
  rgba: Float4
  geom_type: str = "box"

  @property
  def full_dimensions(self) -> Float3:
    """Return full box dimensions; MJCF stores box half-extents in ``size``."""
    return tuple(2.0 * value for value in self.size)  # type: ignore[return-value]


def parse_float_tuple(value: str, *, count: int, attr_name: str) -> tuple[float, ...]:
  parts = value.split()
  if len(parts) != count:
    raise ValueError(
      f"{attr_name!r} must contain {count} floats, got {len(parts)}: {value!r}"
    )
  return tuple(float(part) for part in parts)


def format_float(value: float) -> str:
  return f"{float(value):.6g}"


def format_float_tuple(values: Iterable[float]) -> str:
  return " ".join(format_float(value) for value in values)


def _as_float_tuple(
  values: Sequence[float],
  *,
  count: int,
  field_name: str,
) -> tuple[float, ...]:
  if len(values) != count:
    raise ValueError(f"{field_name} must have {count} values, got {len(values)}")
  return tuple(float(value) for value in values)


def _validate_positive(values: Sequence[float], *, field_name: str) -> None:
  if any(value <= 0.0 for value in values):
    raise ValueError(f"{field_name} values must be positive")


def full_dimensions_to_mjcf_size(full_dimensions: Sequence[float]) -> Float3:
  dimensions = _as_float_tuple(
    full_dimensions,
    count=3,
    field_name="full_dimensions",
  )
  _validate_positive(dimensions, field_name="full_dimensions")
  return tuple(value / 2.0 for value in dimensions)  # type: ignore[return-value]


def mjcf_size_to_full_dimensions(size: Sequence[float]) -> Float3:
  half_extents = _as_float_tuple(size, count=3, field_name="size")
  _validate_positive(half_extents, field_name="size")
  return tuple(2.0 * value for value in half_extents)  # type: ignore[return-value]


class ParkourSceneDocument:
  """Mutable XML document wrapper for parkour terrain box modules."""

  def __init__(
    self,
    *,
    path: Path,
    tree: ET.ElementTree,
    body_name: str = DEFAULT_BODY_NAME,
  ) -> None:
    self.path = Path(path)
    self.tree = tree
    self.body_name = body_name
    self.root = self.tree.getroot()
    self.body = self._find_body(body_name)

  @classmethod
  def from_path(
    cls,
    path: Path | str = DEFAULT_SCENE_PATH,
    *,
    body_name: str = DEFAULT_BODY_NAME,
  ) -> "ParkourSceneDocument":
    scene_path = Path(path).expanduser()
    return cls(path=scene_path, tree=ET.parse(scene_path), body_name=body_name)

  def _find_body(self, body_name: str) -> ET.Element:
    for body in self.root.findall(".//body"):
      if body.attrib.get("name") == body_name:
        return body
    raise ValueError(f"terrain body {body_name!r} was not found in {self.path}")

  def _geom_elements(self) -> tuple[ET.Element, ...]:
    return tuple(
      geom
      for geom in self.body.findall("geom")
      if geom.attrib.get("type", "box") == "box" and "name" in geom.attrib
    )

  def _geom_map(self) -> dict[str, ET.Element]:
    return {str(geom.attrib["name"]): geom for geom in self._geom_elements()}

  def module_names(self) -> tuple[str, ...]:
    return tuple(str(geom.attrib["name"]) for geom in self._geom_elements())

  def modules(self) -> tuple[SceneModule, ...]:
    return tuple(self._module_from_element(geom) for geom in self._geom_elements())

  def get_module(self, name: str) -> SceneModule:
    geom = self._require_geom(name)
    return self._module_from_element(geom)

  def _module_from_element(self, geom: ET.Element) -> SceneModule:
    name = str(geom.attrib["name"])
    pos = parse_float_tuple(geom.attrib.get("pos", "0 0 0"), count=3, attr_name="pos")
    size = parse_float_tuple(
      geom.attrib.get("size", "0 0 0"),
      count=3,
      attr_name="size",
    )
    rgba = parse_float_tuple(
      geom.attrib.get("rgba", "0.6 0.6 0.6 1"),
      count=4,
      attr_name="rgba",
    )
    _validate_positive(size, field_name=f"{name}.size")
    return SceneModule(
      name=name,
      pos=pos,  # type: ignore[arg-type]
      size=size,  # type: ignore[arg-type]
      rgba=rgba,  # type: ignore[arg-type]
      geom_type=str(geom.attrib.get("type", "box")),
    )

  def _require_geom(self, name: str) -> ET.Element:
    geom = self._geom_map().get(name)
    if geom is None:
      raise KeyError(f"terrain module {name!r} was not found")
    return geom

  def add_module(
    self,
    name: str,
    *,
    pos: Sequence[float] = (0.0, 0.0, 0.05),
    size: Sequence[float] | None = None,
    full_dimensions: Sequence[float] | None = None,
    rgba: Sequence[float] = (0.62, 0.42, 0.28, 1.0),
    keep_bottom: bool = True,
  ) -> SceneModule:
    if not name:
      raise ValueError("module name must not be empty")
    if name in self._geom_map():
      raise ValueError(f"terrain module {name!r} already exists")

    mjcf_size = self._resolve_size(size=size, full_dimensions=full_dimensions)
    resolved_pos = _as_float_tuple(pos, count=3, field_name="pos")
    if keep_bottom:
      resolved_pos = (resolved_pos[0], resolved_pos[1], mjcf_size[2])
    resolved_rgba = _as_float_tuple(rgba, count=4, field_name="rgba")

    geom = ET.Element(
      "geom",
      {
        "name": name,
        "type": "box",
        "pos": format_float_tuple(resolved_pos),
        "size": format_float_tuple(mjcf_size),
        "rgba": format_float_tuple(resolved_rgba),
      },
    )
    self.body.append(geom)
    return self.get_module(name)

  def copy_module(
    self,
    source_name: str,
    new_name: str,
    *,
    offset: Sequence[float] = (0.7, 0.0, 0.0),
  ) -> SceneModule:
    source = self.get_module(source_name)
    delta = _as_float_tuple(offset, count=3, field_name="offset")
    return self.add_module(
      new_name,
      pos=tuple(source.pos[index] + delta[index] for index in range(3)),
      size=source.size,
      rgba=source.rgba,
      keep_bottom=False,
    )

  def update_module(
    self,
    name: str,
    *,
    pos: Sequence[float] | None = None,
    size: Sequence[float] | None = None,
    full_dimensions: Sequence[float] | None = None,
    rgba: Sequence[float] | None = None,
    keep_bottom: bool = False,
  ) -> SceneModule:
    geom = self._require_geom(name)
    current = self._module_from_element(geom)
    resolved_size = self._resolve_size(
      size=size,
      full_dimensions=full_dimensions,
      default=current.size,
    )
    resolved_pos = (
      _as_float_tuple(pos, count=3, field_name="pos")
      if pos is not None
      else current.pos
    )
    if keep_bottom:
      bottom_z = current.pos[2] - current.size[2]
      resolved_pos = (resolved_pos[0], resolved_pos[1], bottom_z + resolved_size[2])

    geom.attrib["pos"] = format_float_tuple(resolved_pos)
    geom.attrib["size"] = format_float_tuple(resolved_size)
    if rgba is not None:
      geom.attrib["rgba"] = format_float_tuple(
        _as_float_tuple(rgba, count=4, field_name="rgba")
      )
    return self.get_module(name)

  def _resolve_size(
    self,
    *,
    size: Sequence[float] | None,
    full_dimensions: Sequence[float] | None,
    default: Sequence[float] = (0.18, 0.72, 0.04),
  ) -> Float3:
    if size is not None and full_dimensions is not None:
      raise ValueError("provide only one of size or full_dimensions")
    if full_dimensions is not None:
      return full_dimensions_to_mjcf_size(full_dimensions)
    resolved = _as_float_tuple(size or default, count=3, field_name="size")
    _validate_positive(resolved, field_name="size")
    return resolved  # type: ignore[return-value]

  def delete_module(self, name: str) -> None:
    geom = self._require_geom(name)
    self.body.remove(geom)

  def default_new_name(self, prefix: str = "parkour_custom_box") -> str:
    names = set(self.module_names())
    index = 1
    while True:
      candidate = f"{prefix}_{index:02d}"
      if candidate not in names:
        return candidate
      index += 1

  def save(self, path: Path | str | None = None, *, backup: bool = False) -> Path:
    output_path = Path(path).expanduser() if path is not None else self.path
    if backup and output_path.exists():
      shutil.copy2(output_path, output_path.with_suffix(output_path.suffix + ".bak"))
    ET.indent(self.tree, space="  ")
    self.tree.write(output_path, encoding="unicode", xml_declaration=False)
    return output_path


def add_scene_modules_to_mujoco_spec(
  spec: object,
  *,
  scene_path: Path | str = DEFAULT_SCENE_PATH,
  body_name: str = DEFAULT_BODY_NAME,
) -> None:
  """Append editable scene XML modules to a MuJoCo ``MjSpec`` worldbody."""
  import mujoco

  document = ParkourSceneDocument.from_path(scene_path, body_name=body_name)
  for module in document.modules():
    spec.worldbody.add_geom(
      name=module.name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=list(module.pos),
      size=list(module.size),
      rgba=list(module.rgba),
    )
