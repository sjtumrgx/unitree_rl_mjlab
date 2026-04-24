from pathlib import Path

FORBIDDEN = (
  "Unitree-G1-" + "Topology" + "GetUp",
  "Topology" + "GetUp",
  "topology" + "_getup",
  "g1_" + "topology" + "_getup",
)
ROOTS = [Path("src"), Path("scripts"), Path("tests"), Path("deploy"), Path("doc")]
FILES = [Path("README.md"), Path("README_zh.md")]


def iter_active_files():
  for root in ROOTS:
    if not root.exists():
      continue
    for path in root.rglob("*"):
      if path.is_file() and "__pycache__" not in path.parts and "build" not in path.parts:
        yield path
  for path in FILES:
    if path.exists():
      yield path


def test_active_surfaces_do_not_reference_removed_legacy_getup_names() -> None:
  offenders: list[str] = []
  for path in iter_active_files():
    try:
      text = path.read_text(errors="ignore")
    except UnicodeDecodeError:
      continue
    for needle in FORBIDDEN:
      if needle in text:
        offenders.append(f"{path}: {needle}")
  assert offenders == []
