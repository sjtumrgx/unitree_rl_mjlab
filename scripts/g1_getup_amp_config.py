"""Shared config helpers for the G1 GetUp AMP data workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_PATH = Path("data/g1_getup_amp.yaml")


@dataclass(frozen=True)
class SourceMetadata:
  source_url: str | None
  source_revision: str | None
  source_license: str | None
  upstream_license: str | None


def repo_path(path: str | Path) -> Path:
  """Resolve a user/config path relative to the repository root."""
  resolved = Path(path).expanduser()
  return resolved if resolved.is_absolute() else (_REPO_ROOT / resolved)


def load_workflow_config(config_path: str | Path | None) -> dict[str, Any]:
  """Load the small YAML workflow config.

  PyYAML is used when available.  A tiny YAML-subset parser is kept as fallback
  so the project does not need a new runtime dependency just to read this file.
  """
  if config_path is None:
    return {}
  path = repo_path(config_path)
  if not path.exists():
    if Path(config_path) == DEFAULT_CONFIG_PATH:
      return {}
    raise FileNotFoundError(f"G1 GetUp AMP config does not exist: {path}")
  text = path.read_text()
  try:
    import yaml  # type: ignore
  except ModuleNotFoundError:
    loaded = _parse_simple_yaml(text)
  else:
    loaded = yaml.safe_load(text) or {}
  if not isinstance(loaded, dict):
    raise ValueError(f"G1 GetUp AMP config must be a YAML mapping: {path}")
  return loaded


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
  value = config.get(name, {})
  if value is None:
    return {}
  if not isinstance(value, dict):
    raise ValueError(f"Config section {name!r} must be a mapping")
  return dict(value)


def path_list(value: Any) -> list[Path]:
  if value is None:
    return []
  if isinstance(value, (str, Path)):
    return [Path(value)]
  if isinstance(value, Sequence):
    return [Path(str(item)) for item in value]
  raise ValueError(f"Expected a path or list of paths, got {type(value).__name__}")


def collect_sequence_paths(inputs: Iterable[str | Path]) -> list[Path]:
  """Resolve files/directories into a stable list of candidate motion files."""
  sequence_paths: list[Path] = []
  for item in inputs:
    path = repo_path(item)
    if not path.exists():
      raise FileNotFoundError(f"AMP input path does not exist: {path}")
    if path.is_dir():
      sequence_paths.extend(sorted([*path.rglob("*.pkl"), *path.rglob("*.npz")]))
    else:
      sequence_paths.append(path)
  if not sequence_paths:
    raise FileNotFoundError("No .pkl or .npz motion files were found in configured AMP inputs")
  return sequence_paths


def source_metadata_from_config(
  config: Mapping[str, Any],
  *,
  default_source_url: str | None,
  default_source_license: str | None,
  default_upstream_license: str | None,
  cli_source_url: str | None = None,
  cli_source_revision: str | None = None,
  cli_source_license: str | None = None,
  cli_upstream_license: str | None = None,
) -> SourceMetadata:
  dataset = section(config, "dataset")
  configured_revision = dataset.get("source_revision") or dataset.get(
    "dataset_commit_or_snapshot_id"
  )
  return SourceMetadata(
    source_url=cli_source_url
    if cli_source_url is not None
    else dataset.get("source_url", default_source_url),
    source_revision=cli_source_revision
    if cli_source_revision is not None
    else configured_revision,
    source_license=cli_source_license
    if cli_source_license is not None
    else dataset.get("source_license", default_source_license),
    upstream_license=cli_upstream_license
    if cli_upstream_license is not None
    else dataset.get("upstream_license", default_upstream_license),
  )


def _parse_simple_yaml(text: str) -> dict[str, Any]:
  lines: list[tuple[int, str]] = []
  for raw_line in text.splitlines():
    if not raw_line.strip() or raw_line.lstrip().startswith("#"):
      continue
    content = raw_line.rstrip()
    indent = len(content) - len(content.lstrip(" "))
    lines.append((indent, content.lstrip(" ")))
  if not lines:
    return {}
  parsed, index = _parse_yaml_block(lines, 0, lines[0][0])
  if index != len(lines):
    raise ValueError("Could not parse complete YAML config")
  if not isinstance(parsed, dict):
    raise ValueError("Top-level YAML config must be a mapping")
  return parsed


def _parse_yaml_block(
  lines: list[tuple[int, str]],
  index: int,
  indent: int,
) -> tuple[Any, int]:
  is_list = lines[index][1].startswith("- ")
  if is_list:
    values: list[Any] = []
    while index < len(lines):
      line_indent, content = lines[index]
      if line_indent < indent:
        break
      if line_indent != indent or not content.startswith("- "):
        raise ValueError(f"Invalid YAML list entry: {content}")
      raw = content[2:].strip()
      index += 1
      if raw:
        values.append(_parse_yaml_scalar(raw))
      elif index < len(lines) and lines[index][0] > indent:
        value, index = _parse_yaml_block(lines, index, lines[index][0])
        values.append(value)
      else:
        values.append(None)
    return values, index

  values: dict[str, Any] = {}
  while index < len(lines):
    line_indent, content = lines[index]
    if line_indent < indent:
      break
    if line_indent != indent or ":" not in content:
      raise ValueError(f"Invalid YAML mapping entry: {content}")
    key, raw_value = content.split(":", 1)
    key = key.strip()
    raw_value = raw_value.strip()
    index += 1
    if raw_value:
      values[key] = _parse_yaml_scalar(raw_value)
    elif index < len(lines) and lines[index][0] > indent:
      value, index = _parse_yaml_block(lines, index, lines[index][0])
      values[key] = value
    else:
      values[key] = {}
  return values, index


def _parse_yaml_scalar(raw_value: str) -> Any:
  value = raw_value.strip()
  if (
    (value.startswith('"') and value.endswith('"'))
    or (value.startswith("'") and value.endswith("'"))
  ):
    return value[1:-1]
  lowered = value.lower()
  if lowered in {"true", "false"}:
    return lowered == "true"
  if lowered in {"null", "none", "~"}:
    return None
  try:
    return int(value)
  except ValueError:
    pass
  try:
    return float(value)
  except ValueError:
    return value
