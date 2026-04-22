"""Analysis export helpers for topology-getup policies."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from torch import nn


class _OnnxTopologyAnalysisModel(nn.Module):
  """ONNX-exportable wrapper that emits actions plus topology bottleneck latent."""

  def __init__(self, model, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
    self.topology_bottleneck = copy.deepcopy(model.topology_bottleneck)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

    self.obs_groups_2d = model.obs_groups_2d
    self.obs_dims_2d = model.obs_dims_2d
    self.obs_channels_2d = model.obs_channels_2d
    self.obs_dim_1d = model.obs_dim

  def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent_1d = self.obs_normalizer(obs_1d)
    latent_cnn_list = []
    for i, cnn in enumerate(self.cnns):
      latent_cnn_list.append(cnn(obs_2d[i]))
    latent_cnn = torch.cat(latent_cnn_list, dim=-1)
    topology_latent = self.topology_bottleneck(latent_cnn)
    latent = torch.cat([latent_1d, topology_latent], dim=-1)
    out = self.mlp(latent)
    return self.deterministic_output(out), topology_latent

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    dummy_1d = torch.zeros(1, self.obs_dim_1d)
    dummy_2d = []
    for i in range(len(self.obs_groups_2d)):
      h, w = self.obs_dims_2d[i]
      c = self.obs_channels_2d[i]
      dummy_2d.append(torch.zeros(1, c, h, w))
    return (dummy_1d, *dummy_2d)

  @property
  def input_names(self) -> list[str]:
    return ["obs", *self.obs_groups_2d]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "topology_latent"]


def export_topology_analysis_to_onnx(policy, path: str | os.PathLike[str], filename: str = "policy_analysis.onnx") -> str | None:
  if not hasattr(policy, "topology_bottleneck"):
    return None
  model = _OnnxTopologyAnalysisModel(policy)
  model.to("cpu")
  model.eval()
  path = str(path)
  os.makedirs(path, exist_ok=True)
  save_path = os.path.join(path, filename)
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    save_path,
    export_params=True,
    opset_version=18,
    verbose=False,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )
  return save_path
