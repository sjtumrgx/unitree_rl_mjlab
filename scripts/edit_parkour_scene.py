#!/usr/bin/env python3
"""Viser-based editor for the deterministic G1 parkour MJCF terrain modules."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import webbrowser
from typing import Any, Sequence

from src.parkour.scene_editor import (
  DEFAULT_BODY_NAME,
  DEFAULT_SCENE_PATH,
  ParkourSceneDocument,
  SceneModule,
)


NO_MODULE = "<no editable modules>"


def _tuple3(values: Sequence[float]) -> tuple[float, float, float]:
  return tuple(float(value) for value in values[:3])  # type: ignore[return-value]


def _rgba_float_to_u8(rgba: Sequence[float]) -> tuple[int, int, int, int]:
  return tuple(
    max(0, min(255, int(round(float(value) * 255.0))))
    for value in rgba[:4]
  )  # type: ignore[return-value]


def _rgba_u8_to_float(rgba: Sequence[int]) -> tuple[float, float, float, float]:
  return tuple(float(value) / 255.0 for value in rgba[:4])  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Open a browser-based Viser editor for the G1 parkour MJCF terrain. "
      "Edits are applied to the XML in memory and written only when Save is "
      "clicked."
    )
  )
  parser.add_argument(
    "--scene",
    type=Path,
    default=DEFAULT_SCENE_PATH,
    help="MJCF scene XML to edit.",
  )
  parser.add_argument(
    "--body",
    default=DEFAULT_BODY_NAME,
    help="Terrain body whose direct box geoms should be editable.",
  )
  parser.add_argument("--host", default="127.0.0.1", help="Viser host.")
  parser.add_argument("--port", type=int, default=8080, help="Viser port.")
  parser.add_argument(
    "--no-backup",
    action="store_true",
    help="Do not create <scene>.bak when saving.",
  )
  parser.add_argument(
    "--open-browser",
    action="store_true",
    help="Open the editor URL in the default browser after startup.",
  )
  return parser


class ParkourSceneViserEditor:
  """Small Viser app for creating, deleting, moving, and resizing box geoms."""

  def __init__(self, args: argparse.Namespace) -> None:
    import viser

    self.args = args
    self.scene_path = args.scene.expanduser()
    self.document = ParkourSceneDocument.from_path(
      self.scene_path,
      body_name=args.body,
    )
    self.server = viser.ViserServer(
      host=args.host,
      port=args.port,
      label="G1 Parkour Scene Editor",
    )
    self.selected_name: str | None = None
    self.box_handles: dict[str, Any] = {}
    self.label_handles: dict[str, Any] = {}
    self.editor_handles: list[Any] = []
    self.position_control: Any | None = None
    self._syncing_gui = False

    self.server.scene.add_grid(
      "/ground_grid",
      width=30.0,
      height=4.0,
      plane="xy",
      cell_size=0.25,
      section_size=1.0,
      position=(12.5, 0.0, 0.0),
      plane_opacity=0.03,
    )
    self.transform = self.server.scene.add_transform_controls(
      "/selected_transform",
      visible=False,
      disable_rotations=True,
      scale=0.5,
    )
    self._build_gui()
    self._rebuild_scene()
    self._select_first_module()
    self._wire_transform_controls()

  def _build_gui(self) -> None:
    names = self._selector_options()
    self.status = self.server.gui.add_markdown(
      self._status_text("Loaded scene. Click a module or use the dropdown.")
    )
    self.selector = self.server.gui.add_dropdown(
      "Module",
      names,
      initial_value=names[0],
    )
    self.show_labels = self.server.gui.add_checkbox(
      "Show labels",
      initial_value=True,
    )
    self.keep_bottom = self.server.gui.add_checkbox(
      "Keep bottom fixed when resizing",
      initial_value=True,
      hint=(
        "When full dimensions change, keep the old bottom z and update center z."
      ),
    )

    self.editor_folder = self.server.gui.add_folder("Selected module")
    self.add_folder = self.server.gui.add_folder("Add / delete / save")
    with self.add_folder:
      self.new_name = self.server.gui.add_text(
        "New module name",
        initial_value=self.document.default_new_name(),
      )
      self.add_button = self.server.gui.add_button("Add default box")
      self.duplicate_button = self.server.gui.add_button("Duplicate selected")
      self.delete_button = self.server.gui.add_button("Delete selected")
      self.save_button = self.server.gui.add_button("Save XML")
      self.reload_button = self.server.gui.add_button("Reload from disk")

    @self.selector.on_update
    def _(_: Any) -> None:
      if self._syncing_gui:
        return
      if self.selector.value != NO_MODULE:
        self.select_module(str(self.selector.value))

    @self.show_labels.on_update
    def _(_: Any) -> None:
      for label in self.label_handles.values():
        label.visible = bool(self.show_labels.value)

    @self.add_button.on_click
    def _(_: Any) -> None:
      self._add_default_module()

    @self.duplicate_button.on_click
    def _(_: Any) -> None:
      self._duplicate_selected_module()

    @self.delete_button.on_click
    def _(_: Any) -> None:
      self._delete_selected_module()

    @self.save_button.on_click
    def _(_: Any) -> None:
      self._save_scene()

    @self.reload_button.on_click
    def _(_: Any) -> None:
      self._reload_scene()

  def _wire_transform_controls(self) -> None:
    @self.transform.on_update
    def _(_: Any) -> None:
      if self._syncing_gui or self.selected_name is None:
        return
      module = self.document.update_module(
        self.selected_name,
        pos=_tuple3(self.transform.position),
      )
      self._redraw_module(module.name)
      self._sync_position_control(module)
      self._set_status(f"Moved {module.name}: pos={module.pos}")

  def _selector_options(self) -> tuple[str, ...]:
    names = self.document.module_names()
    return names if names else (NO_MODULE,)

  def _status_text(self, message: str) -> str:
    return (
      f"**Scene:** `{self.scene_path}`  \n"
      f"**Terrain body:** `{self.args.body}`  \n"
      f"**Status:** {message}"
    )

  def _set_status(self, message: str) -> None:
    self.status.content = self._status_text(message)

  def _select_first_module(self) -> None:
    options = self._selector_options()
    if options[0] != NO_MODULE:
      self.select_module(options[0])
    else:
      self._build_selected_module_controls(None)

  def _refresh_selector(self) -> None:
    options = self._selector_options()
    self._syncing_gui = True
    try:
      self.selector.options = options
      if self.selected_name not in options:
        self.selected_name = options[0] if options[0] != NO_MODULE else None
      self.selector.value = self.selected_name or NO_MODULE
    finally:
      self._syncing_gui = False

  def _rebuild_scene(self) -> None:
    for handle in [*self.box_handles.values(), *self.label_handles.values()]:
      handle.remove()
    self.box_handles.clear()
    self.label_handles.clear()
    for module in self.document.modules():
      self._draw_module(module)

  def _draw_module(self, module: SceneModule) -> None:
    color = _rgba_float_to_u8(module.rgba)[:3]
    opacity = module.rgba[3] if module.rgba[3] < 0.999 else None
    handle = self.server.scene.add_box(
      f"/terrain/{module.name}",
      color=color,
      dimensions=module.full_dimensions,
      position=module.pos,
      opacity=opacity,
      side="double",
    )

    @handle.on_click
    def _(_: Any, module_name: str = module.name) -> None:
      self.select_module(module_name)

    label = self.server.scene.add_label(
      f"/terrain_labels/{module.name}",
      text=module.name,
      position=(module.pos[0], module.pos[1], module.pos[2] + module.size[2] + 0.05),
      visible=bool(self.show_labels.value),
      font_size_mode="scene",
      font_scene_height=0.09,
      anchor="bottom-center",
    )
    self.box_handles[module.name] = handle
    self.label_handles[module.name] = label

  def _redraw_module(self, name: str) -> None:
    if name in self.box_handles:
      self.box_handles.pop(name).remove()
    if name in self.label_handles:
      self.label_handles.pop(name).remove()
    self._draw_module(self.document.get_module(name))

  def select_module(self, name: str) -> None:
    if name not in self.document.module_names():
      return
    self.selected_name = name
    module = self.document.get_module(name)
    self._syncing_gui = True
    try:
      self.selector.value = name
      self.transform.visible = True
      self.transform.position = module.pos
    finally:
      self._syncing_gui = False
    self._build_selected_module_controls(module)
    self._set_status(f"Selected {name}.")

  def _build_selected_module_controls(self, module: SceneModule | None) -> None:
    for handle in self.editor_handles:
      handle.remove()
    self.editor_handles.clear()
    self.position_control = None

    with self.editor_folder:
      if module is None:
        empty = self.server.gui.add_markdown("No editable module selected.")
        self.editor_handles.append(empty)
        self.transform.visible = False
        return

      position = self.server.gui.add_vector3(
        "Position center (m)",
        module.pos,
        step=0.01,
      )
      dimensions = self.server.gui.add_vector3(
        "Full dimensions L/W/H (m)",
        module.full_dimensions,
        min=(0.01, 0.01, 0.001),
        step=0.01,
      )
      rgba = self.server.gui.add_rgba(
        "Color RGBA",
        _rgba_float_to_u8(module.rgba),
      )
      self.editor_handles.extend([position, dimensions, rgba])
      self.position_control = position

      @position.on_update
      def _(_: Any, module_name: str = module.name) -> None:
        if self._syncing_gui:
          return
        updated = self.document.update_module(
          module_name,
          pos=_tuple3(position.value),
        )
        self._redraw_module(module_name)
        self._sync_transform(updated)
        self._set_status(f"Updated {module_name} position.")

      @dimensions.on_update
      def _(_: Any, module_name: str = module.name) -> None:
        if self._syncing_gui:
          return
        updated = self.document.update_module(
          module_name,
          full_dimensions=_tuple3(dimensions.value),
          keep_bottom=bool(self.keep_bottom.value),
        )
        self._redraw_module(module_name)
        self._sync_transform(updated)
        self._sync_position_control(updated)
        self._set_status(f"Updated {module_name} dimensions.")

      @rgba.on_update
      def _(_: Any, module_name: str = module.name) -> None:
        if self._syncing_gui:
          return
        updated = self.document.update_module(
          module_name,
          rgba=_rgba_u8_to_float(rgba.value),
        )
        self._redraw_module(module_name)
        self._sync_transform(updated)
        self._set_status(f"Updated {module_name} color.")

  def _sync_transform(self, module: SceneModule) -> None:
    self._syncing_gui = True
    try:
      self.transform.position = module.pos
    finally:
      self._syncing_gui = False

  def _sync_position_control(self, module: SceneModule) -> None:
    if self.position_control is None:
      return
    self._syncing_gui = True
    try:
      self.position_control.value = module.pos
    finally:
      self._syncing_gui = False

  def _candidate_new_name(self) -> str:
    name = str(self.new_name.value).strip()
    return name or self.document.default_new_name()

  def _add_default_module(self) -> None:
    try:
      existing = self.document.modules()
      next_x = max((module.pos[0] for module in existing), default=0.0) + 0.7
      module = self.document.add_module(
        self._candidate_new_name(),
        pos=(next_x, 0.0, 0.0),
        full_dimensions=(0.36, 1.44, 0.08),
        keep_bottom=True,
      )
    except Exception as exc:  # noqa: BLE001 - GUI should report recoverable edits.
      self._set_status(f"Add failed: {exc}")
      return
    self._draw_module(module)
    self._after_module_collection_changed(module.name)
    self._set_status(f"Added {module.name}. Click Save XML to persist.")

  def _duplicate_selected_module(self) -> None:
    if self.selected_name is None:
      self._set_status("Duplicate failed: no module selected.")
      return
    try:
      module = self.document.copy_module(
        self.selected_name,
        self._candidate_new_name(),
        offset=(0.7, 0.0, 0.0),
      )
    except Exception as exc:  # noqa: BLE001 - GUI should report recoverable edits.
      self._set_status(f"Duplicate failed: {exc}")
      return
    self._draw_module(module)
    self._after_module_collection_changed(module.name)
    self._set_status(f"Duplicated {self.selected_name} as {module.name}.")

  def _delete_selected_module(self) -> None:
    if self.selected_name is None:
      self._set_status("Delete failed: no module selected.")
      return
    deleted = self.selected_name
    try:
      self.document.delete_module(deleted)
    except Exception as exc:  # noqa: BLE001 - GUI should report recoverable edits.
      self._set_status(f"Delete failed: {exc}")
      return
    if deleted in self.box_handles:
      self.box_handles.pop(deleted).remove()
    if deleted in self.label_handles:
      self.label_handles.pop(deleted).remove()
    self._after_module_collection_changed(None)
    self._set_status(f"Deleted {deleted}. Click Save XML to persist.")

  def _after_module_collection_changed(self, preferred_selection: str | None) -> None:
    self.selected_name = preferred_selection
    self._refresh_selector()
    if self.selected_name is not None:
      self.select_module(self.selected_name)
    else:
      self._build_selected_module_controls(None)
      self.transform.visible = False
    self._syncing_gui = True
    try:
      self.new_name.value = self.document.default_new_name()
    finally:
      self._syncing_gui = False

  def _save_scene(self) -> None:
    try:
      self.document.save(backup=not self.args.no_backup)
    except Exception as exc:  # noqa: BLE001 - GUI should report recoverable edits.
      self._set_status(f"Save failed: {exc}")
      return
    suffix = "" if self.args.no_backup else " Backup written to .xml.bak."
    self._set_status(f"Saved XML.{suffix}")

  def _reload_scene(self) -> None:
    try:
      self.document = ParkourSceneDocument.from_path(
        self.scene_path,
        body_name=self.args.body,
      )
    except Exception as exc:  # noqa: BLE001 - GUI should report recoverable edits.
      self._set_status(f"Reload failed: {exc}")
      return
    self.selected_name = None
    self._refresh_selector()
    self._rebuild_scene()
    self._select_first_module()
    self._set_status("Reloaded scene from disk.")

  def run(self) -> None:
    url = f"http://{self.args.host}:{self.args.port}"
    print(f"[parkour-scene-editor] Open {url}")
    print("[parkour-scene-editor] Ctrl-C to stop. Use Save XML to persist edits.")
    if self.args.open_browser:
      webbrowser.open(url)
    try:
      while True:
        time.sleep(0.5)
    except KeyboardInterrupt:
      print("\n[parkour-scene-editor] stopped")


def main() -> None:
  args = build_parser().parse_args()
  ParkourSceneViserEditor(args).run()


if __name__ == "__main__":
  main()
