from pathlib import Path


def test_host_migration_mapping_doc_covers_required_sources() -> None:
  doc = Path("doc/g1_getup_host_migration.md")
  assert doc.exists()
  text = doc.read_text()
  for needle in (
    "g1_config_ground.py",
    "g1_config_platform.py",
    "g1_config_wall.py",
    "g1_config_slope.py",
    "host_ground.py",
    "host_platform.py",
    "host_wall.py",
    "host_slope.py",
    "g1_utils.py",
    "g1_ground_prone",
    "src/tasks/velocity/config/g1_getup/env_cfgs.py",
    "src/tasks/velocity/mdp/getup/",
  ):
    assert needle in text
