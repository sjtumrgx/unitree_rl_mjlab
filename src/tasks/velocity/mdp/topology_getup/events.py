from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp.anti_fall.events import reset_antifall_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_PRESETS: tuple[dict[str, object], ...] = (
  {
    "name": "supine",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (0.0, 0.05),
      "roll": (math.pi - 0.3, math.pi + 0.3),
      "pitch": (-0.3, 0.3),
      "yaw": (-math.pi, math.pi),
    },
  },
  {
    "name": "left_side",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (0.0, 0.05),
      "roll": (math.pi / 2 - 0.25, math.pi / 2 + 0.25),
      "pitch": (-0.35, 0.35),
      "yaw": (-math.pi, math.pi),
    },
  },
  {
    "name": "right_side",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (0.0, 0.05),
      "roll": (-math.pi / 2 - 0.25, -math.pi / 2 + 0.25),
      "pitch": (-0.35, 0.35),
      "yaw": (-math.pi, math.pi),
    },
  },
  {
    "name": "seated_fall",
    "pose_range": {
      "x": (-0.15, 0.15),
      "y": (-0.15, 0.15),
      "z": (0.18, 0.28),
      "roll": (-0.2, 0.2),
      "pitch": (math.pi / 2 - 0.35, math.pi / 2 + 0.35),
      "yaw": (-math.pi, math.pi),
    },
  },
)
_DEFAULT_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.25, 0.25),
  "roll": (-0.75, 0.75),
  "pitch": (-0.75, 0.75),
  "yaw": (-0.75, 0.75),
}


def reset_root_state_from_presets(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  presets: Sequence[dict[str, object]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    ids = env_ids.to(device=env.device, dtype=torch.long)
  if ids.numel() == 0:
    return

  reset_antifall_state(env, ids)
  preset_list = tuple(_DEFAULT_PRESETS if presets is None else presets)
  vel_range = _DEFAULT_VELOCITY_RANGE if velocity_range is None else velocity_range
  preset_indices = torch.randint(len(preset_list), (ids.numel(),), device=env.device)
  for preset_idx, preset in enumerate(preset_list):
    selected = ids[preset_indices == preset_idx]
    if selected.numel() == 0:
      continue
    envs_mdp.reset_root_state_uniform(
      env,
      selected,
      pose_range=preset["pose_range"],
      velocity_range=vel_range,
      asset_cfg=asset_cfg,
    )
