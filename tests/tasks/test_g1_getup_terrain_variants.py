import pytest

from src.tasks.velocity.config.g1_getup.env_cfgs import (
  GETUP_TERRAIN_VARIANTS,
  HOST_TERRAIN_PARITY,
  unitree_g1_getup_env_cfg,
)


@pytest.mark.parametrize("terrain", GETUP_TERRAIN_VARIANTS)
def test_host_terrain_variants_are_instantiable(terrain: str) -> None:
  cfg = unitree_g1_getup_env_cfg(terrain=terrain)
  parity = HOST_TERRAIN_PARITY[terrain]
  generator = cfg.scene.terrain.terrain_generator

  assert getattr(cfg, "getup_terrain") == terrain
  assert getattr(cfg, "host_source_task") == f"g1_{terrain}"
  assert getattr(cfg, "host_parity") == parity
  assert generator.num_rows == parity["num_rows"]
  assert generator.num_cols == parity["num_cols"]
  assert cfg.events["getup_assist_force"].params["force_n"] == parity["pull_force_n"]


def test_prone_variant_is_not_exposed() -> None:
  assert "ground_prone" not in GETUP_TERRAIN_VARIANTS
  with pytest.raises(ValueError):
    unitree_g1_getup_env_cfg(terrain="ground_prone")
