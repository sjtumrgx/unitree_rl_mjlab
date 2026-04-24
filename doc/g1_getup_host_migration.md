# Unitree G1 HoST Get-Up Migration

This document maps the HoST IsaacGym/legged_gym G1 standing-up tasks to the mjlab
`Unitree-G1-GetUp` task.

## Scope

Migrated HoST tasks:

| HoST task | mjlab public task | Terrain variant | Status |
| --- | --- | --- | --- |
| `g1_ground` | `Unitree-G1-GetUp` | `ground` | migrated |
| `g1_platform` | `Unitree-G1-GetUp` | `platform` | migrated |
| `g1_wall` | `Unitree-G1-GetUp` | `wall` | migrated |
| `g1_slope` | `Unitree-G1-GetUp` | `slope` | migrated |
| `g1_ground_prone` | none | none | deferred: out of first-pass scope |

## Source-to-destination map

| HoST source | Destination / disposition |
| --- | --- |
| `HoST/legged_gym/legged_gym/envs/__init__.py` | Source task inventory for the four migrated variants; `g1_ground_prone` marked deferred. |
| `HoST/legged_gym/legged_gym/envs/g1/g1_config_ground.py` | `src/tasks/velocity/config/g1_getup/env_cfgs.py` `HOST_TERRAIN_PARITY["ground"]`; RL run name in `rl_cfg.py`. |
| `HoST/legged_gym/legged_gym/envs/g1/g1_config_platform.py` | `HOST_TERRAIN_PARITY["platform"]`; terrain rows/cols/proportions and height thresholds preserved as metadata/config. |
| `HoST/legged_gym/legged_gym/envs/g1/g1_config_wall.py` | `HOST_TERRAIN_PARITY["wall"]`; wall terrain dimensions and pull force recorded. |
| `HoST/legged_gym/legged_gym/envs/g1/g1_config_slope.py` | `HOST_TERRAIN_PARITY["slope"]`; slope-specific lower phase height thresholds recorded. |
| `HoST/legged_gym/legged_gym/envs/base/host_ground.py` | Reward/reset/termination/progress behavior mapped to `src/tasks/velocity/mdp/getup/` plus env reward/event wiring in `g1_getup/env_cfgs.py`. |
| `HoST/legged_gym/legged_gym/envs/base/host_platform.py` | Same mjlab MDP destination; platform-specific terrain/config differences captured by terrain variant metadata. |
| `HoST/legged_gym/legged_gym/envs/base/host_wall.py` | Same mjlab MDP destination; wall-specific terrain/config differences captured by terrain variant metadata. |
| `HoST/legged_gym/legged_gym/envs/base/host_slope.py` | Same mjlab MDP destination; slope-specific height thresholds captured by terrain variant metadata. |
| `HoST/legged_gym/legged_gym/envs/g1/g1_utils.py` | Shared G1 conventions reviewed; mjlab uses existing `src/assets/robots` G1 constants and `SceneEntityCfg` names instead of importing IsaacGym helpers. |

## Parity notes

- Terrain variants are first-class metadata on the mjlab env cfg: `getup_terrain`,
  `host_source_task`, and `host_parity`.
- Pull/assist force is represented by `getup_assist_force` in the mjlab event graph.
- HoST reward groups are represented by mjlab reward terms in `g1_getup/env_cfgs.py`
  and MDP functions in `src/tasks/velocity/mdp/getup/`.
- IsaacGym tensor APIs are not imported; equivalent mjlab `ManagerBasedRlEnv` access is used.
- Long training convergence is not claimed by this migration. Import/config tests and
  optional reset/step smoke are the first-pass verification gates.

## Deferred / rejected items

- HoST `g1_ground_prone` is deferred by requirement.
- Legacy depth-specific distillation, bottleneck, and artifact promotion flows
  were removed rather than retained as aliases.
