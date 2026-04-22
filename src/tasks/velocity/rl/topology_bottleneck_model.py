"""CNN model with an explicit topology bottleneck latent for topology-getup tasks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.rl.spatial_softmax import SpatialSoftmaxCNN


class TopologyBottleneckCNNModel(CNNModel):
  """Depth-conditioned CNN model with an explicit bottleneck over 2D geometry latents.

  This keeps the student observation contract fixed while making the method claim sharper
  than a naive depth-conditioned baseline: the 2D geometry latents are compressed through
  a dedicated bottleneck before being concatenated with 1D proprioceptive observations.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      _cnns = cnns
      bottleneck_dim = None
    else:
      if cnn_cfg is None:
        raise ValueError("CNN configurations must be provided if CNNs are not shared.")
      if not all(isinstance(v, dict) for v in cnn_cfg.values()):
        cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
      if len(cnn_cfg) != len(self.obs_groups_2d):
        raise ValueError(
          "The number of CNN configurations must match the number of 2D observation groups."
        )
      _cnns = {}
      bottleneck_dim = None
      for idx, obs_group in enumerate(self.obs_groups_2d):
        group_cfg = dict(cnn_cfg[obs_group])
        group_cfg.pop("spatial_softmax", None)
        temperature = group_cfg.pop("spatial_softmax_temperature", 1.0)
        local_bottleneck_dim = int(group_cfg.pop("bottleneck_dim", 64))
        if bottleneck_dim is None:
          bottleneck_dim = local_bottleneck_dim
        elif bottleneck_dim != local_bottleneck_dim:
          raise ValueError("All CNN groups must share the same bottleneck_dim.")
        _cnns[obs_group] = SpatialSoftmaxCNN(
          input_dim=self.obs_dims_2d[idx],
          input_channels=self.obs_channels_2d[idx],
          temperature=temperature,
          **group_cfg,
        )

    if bottleneck_dim is None:
      bottleneck_dim = 64

    self._pre_bottleneck_dim = 0
    for cnn in _cnns.values():
      if cnn.output_channels is not None:
        raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
      self._pre_bottleneck_dim += int(cnn.output_dim)  # type: ignore[arg-type]

    self.bottleneck_dim = bottleneck_dim
    self.cnn_latent_dim = self.bottleneck_dim

    MLPModel.__init__(
      self,
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    self.cnns = _cnns if isinstance(_cnns, nn.ModuleDict) else nn.ModuleDict(_cnns)
    self.topology_bottleneck = MLP(
      self._pre_bottleneck_dim,
      self.bottleneck_dim,
      (self.bottleneck_dim,),
      activation,
    )

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    latent_1d = MLPModel.get_latent(self, obs)
    latent_cnn_list = [self.cnns[obs_group](obs[obs_group]) for obs_group in self.obs_groups_2d]
    latent_cnn = torch.cat(latent_cnn_list, dim=-1)
    topology_latent = self.topology_bottleneck(latent_cnn)
    return torch.cat([latent_1d, topology_latent], dim=-1)
