from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "train.py"


def _load_train_helpers():
  source = SCRIPT_PATH.read_text(encoding="utf-8")
  module = ast.parse(source, filename=str(SCRIPT_PATH))
  selected = []
  for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
      "_normalize_gpu_ids_cli_args",
      "_parse_gpu_ids_arg",
    }:
      selected.append(node)

  helper_module = ast.Module(body=selected, type_ignores=[])
  namespace = {"ast": ast, "Literal": Literal}
  exec(compile(helper_module, str(SCRIPT_PATH), "exec"), namespace)
  return (
    namespace["_normalize_gpu_ids_cli_args"],
    namespace["_parse_gpu_ids_arg"],
  )


def test_normalize_gpu_ids_keeps_new_list_form() -> None:
  normalize, _ = _load_train_helpers()

  args = ["--gpu-ids", "[0,1]", "--agent.max-iterations=5"]

  assert normalize(args) == args


def test_normalize_gpu_ids_rewrites_legacy_spaced_form() -> None:
  normalize, _ = _load_train_helpers()

  args = ["--gpu-ids", "0", "1", "--agent.max-iterations=5"]

  assert normalize(args) == [
    "--gpu-ids",
    "[0,1]",
    "--agent.max-iterations=5",
  ]


@pytest.mark.parametrize(
  ("raw_value", "expected"),
  [
    ("[0,1]", [0, 1]),
    ("0,1", [0, 1]),
    ("0", [0]),
    ("all", "all"),
    ("cpu", None),
  ],
)
def test_parse_gpu_ids_accepts_supported_forms(raw_value, expected) -> None:
  _, parse = _load_train_helpers()

  assert parse(raw_value) == expected


def test_parse_gpu_ids_rejects_non_integer_entries() -> None:
  _, parse = _load_train_helpers()

  with pytest.raises(ValueError):
    parse("[0,'bad']")
