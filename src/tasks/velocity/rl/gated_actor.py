"""Gated dual-prior actor for AntiFall-GetUp."""

from __future__ import annotations

import copy
from collections.abc import Sequence
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.modules.rnn import HiddenState
from rsl_rl.utils import unpad_trajectories


def _default_recovery_gate_indices(actor_obs_dim: int) -> tuple[int, int] | tuple[()]:
  """Infer AntiFall-GetUp's latest ``getup_progress`` height/upright columns."""

  # Final AntiFall-GetUp actor layout:
  # base_ang_vel/projected_gravity/command: 3 features x 6 history each
  # joint_pos/joint_vel/actions: 29 features x 6 history each
  # getup_progress: 5 features x 6 history, latest frame starts at +25.
  if actor_obs_dim == 2177:
    return (601, 602, 606)

  if actor_obs_dim == 2176:
    return (601, 602)

  # Early AntiFall-GetUp checkpoints used compact three-frame proprioception
  # plus three-frame getup_progress before BFM state.  Keep this fallback only
  # for local legacy diagnostics; the current task uses the 2176-wide contract.
  if actor_obs_dim == 850:
    return (298, 299)

  return ()


class GatedAntiFallGetUpActor(nn.Module):
  """Select Stage4b walking or GetUp recovery branches from actor observations.

  The task intentionally uses different action semantics in different physical
  phases: default-offset targets while upright and current-pose deltas while
  fallen/recovering.  A single warm-started MLP tends to forget one prior when
  trained on the other.  This actor keeps two MLP priors and uses the same
  fallen/recovery condition as the action term to choose the output per env.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    walking_input_indices: Sequence[int] | None = None,
    recovery_gate_indices: Sequence[int] | None = None,
    recovery_gate_height_threshold: float = 0.55,
    recovery_gate_upright_threshold: float = math.sqrt(max(0.0, 1.0 - 0.75**2)),
    recovery_gate_window_threshold: float = 0.5,
  ) -> None:
    super().__init__()
    if distribution_cfg is None:
      raise ValueError("GatedAntiFallGetUpActor requires a stochastic distribution_cfg.")

    actor_obs_dim = int(obs[obs_groups[obs_set][0]].shape[-1])
    if walking_input_indices is None:
      walking_input_indices = tuple(range(actor_obs_dim))
    if recovery_gate_indices is None:
      recovery_gate_indices = _default_recovery_gate_indices(actor_obs_dim)
    self.register_buffer("walking_input_indices", torch.tensor(tuple(walking_input_indices), dtype=torch.long))
    self.register_buffer("recovery_gate_indices", torch.tensor(tuple(recovery_gate_indices), dtype=torch.long))
    self.recovery_gate_height_threshold = float(recovery_gate_height_threshold)
    self.recovery_gate_upright_threshold = float(recovery_gate_upright_threshold)
    self.recovery_gate_window_threshold = float(recovery_gate_window_threshold)

    dist_cfg = dict(distribution_cfg)
    dist_class_name = dist_cfg.pop("class_name")
    if dist_class_name != "GaussianDistribution":
      raise ValueError("GatedAntiFallGetUpActor currently supports GaussianDistribution only.")
    self.distribution = GaussianDistribution(output_dim, **dist_cfg)

    branch_dist_cfg = {"class_name": "GaussianDistribution", **dist_cfg}
    self.walking_actor = MLPModel(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=branch_dist_cfg.copy(),
    )
    self.recovery_actor = MLPModel(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=branch_dist_cfg.copy(),
    )

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    walking_mean = self.walking_actor(obs)
    recovery_mean = self.recovery_actor(obs)
    recovery_gate = self._recovery_gate(obs).to(dtype=torch.bool, device=walking_mean.device)
    mean = torch.where(recovery_gate.unsqueeze(-1), recovery_mean, walking_mean)
    if stochastic_output:
      self.distribution.update(mean)
      return self.distribution.sample()
    return mean

  def _actor_obs(self, obs: TensorDict) -> torch.Tensor:
    # This model is only used with obs_groups={"actor": ("actor",), ...}.
    return obs[self.walking_actor.obs_groups[0]]

  def _recovery_gate(self, obs: TensorDict) -> torch.Tensor:
    actor_obs = self._actor_obs(obs)
    if self.recovery_gate_indices.numel() < 2:
      return torch.zeros(actor_obs.shape[0], dtype=torch.bool, device=actor_obs.device)
    features = actor_obs.index_select(-1, self.recovery_gate_indices.to(device=actor_obs.device))
    height_progress = features[:, 0]
    facing_up = features[:, 1]
    fallen = (height_progress < self.recovery_gate_height_threshold) | (
      facing_up < self.recovery_gate_upright_threshold
    )
    if features.shape[-1] < 3:
      return fallen
    recovery_window = features[:, 2] > self.recovery_gate_window_threshold
    return fallen | recovery_window

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    return self.walking_actor.get_latent(obs)

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    self.walking_actor.reset(dones=dones, hidden_state=hidden_state)
    self.recovery_actor.reset(dones=dones, hidden_state=hidden_state)

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)

  def update_normalization(self, obs: TensorDict) -> None:
    recovery_gate = self._recovery_gate(obs).to(dtype=torch.bool, device=self._actor_obs(obs).device)
    self._update_branch_normalization(self.walking_actor, obs, ~recovery_gate)
    self._update_branch_normalization(self.recovery_actor, obs, recovery_gate)

  @staticmethod
  def _update_branch_normalization(model: MLPModel, obs: TensorDict, mask: torch.Tensor) -> None:
    """Update a branch normalizer only from samples selected for that branch.

    The walking and GetUp priors come from different physical/action contracts.
    Feeding fallen recovery observations into the walking normalizer, or upright
    walking observations into the recovery normalizer, shifts the warm-started
    input distribution before PPO has learned a useful correction.  Keep each
    branch's statistics aligned with the states where its output is actually
    used.
    """

    if not getattr(model, "obs_normalization", False):
      return
    if mask.numel() == 0 or not bool(mask.any().item()):
      return
    mask = mask.to(device=obs[model.obs_groups[0]].device, dtype=torch.bool)
    obs_list = [obs[obs_group] for obs_group in model.obs_groups]
    mlp_obs = torch.cat(obs_list, dim=-1)
    model.obs_normalizer.update(mlp_obs[mask])

  def as_jit(self) -> nn.Module:
    return _ExportableGatedAntiFallGetUpActor(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _ExportableGatedAntiFallGetUpActor(self, verbose=verbose)


class _ExportableGatedAntiFallGetUpActor(nn.Module):
  """Single-tensor export wrapper for ``GatedAntiFallGetUpActor``."""

  is_recurrent: bool = False

  def __init__(self, model: GatedAntiFallGetUpActor, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.walking_obs_normalizer = copy.deepcopy(model.walking_actor.obs_normalizer)
    self.walking_mlp = copy.deepcopy(model.walking_actor.mlp)
    self.recovery_obs_normalizer = copy.deepcopy(model.recovery_actor.obs_normalizer)
    self.recovery_mlp = copy.deepcopy(model.recovery_actor.mlp)
    self.walking_deterministic_output = (
      model.walking_actor.distribution.as_deterministic_output_module()
      if model.walking_actor.distribution is not None
      else nn.Identity()
    )
    self.recovery_deterministic_output = (
      model.recovery_actor.distribution.as_deterministic_output_module()
      if model.recovery_actor.distribution is not None
      else nn.Identity()
    )
    self.register_buffer("recovery_gate_indices", model.recovery_gate_indices.detach().clone())
    self.recovery_gate_height_threshold = model.recovery_gate_height_threshold
    self.recovery_gate_upright_threshold = model.recovery_gate_upright_threshold
    self.recovery_gate_window_threshold = model.recovery_gate_window_threshold
    self.input_size = int(model.walking_actor.obs_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    walking = self.walking_deterministic_output(self.walking_mlp(self.walking_obs_normalizer(x)))
    recovery = self.recovery_deterministic_output(self.recovery_mlp(self.recovery_obs_normalizer(x)))
    gate = self._recovery_gate(x).to(dtype=torch.bool, device=x.device)
    return torch.where(gate.unsqueeze(-1), recovery, walking)

  def _recovery_gate(self, x: torch.Tensor) -> torch.Tensor:
    if self.recovery_gate_indices.numel() < 2:
      return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    features = x.index_select(-1, self.recovery_gate_indices.to(device=x.device))
    fallen = (features[:, 0] < self.recovery_gate_height_threshold) | (
      features[:, 1] < self.recovery_gate_upright_threshold
    )
    if features.shape[-1] < 3:
      return fallen
    recovery_window = features[:, 2] > self.recovery_gate_window_threshold
    return fallen | recovery_window

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass
