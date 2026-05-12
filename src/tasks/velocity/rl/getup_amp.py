"""AMP PPO extension for the optional ground-only G1 GetUp fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.algorithms import PPO

from .getup_amp_data import AmpExpertDataset


class AmpDiscriminator(nn.Module):
  """Binary discriminator over consecutive AMP observations."""

  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 128)) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(last_dim, int(hidden_dim)))
      layers.append(nn.ELU())
      last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, transitions: torch.Tensor) -> torch.Tensor:
    return self.net(transitions).squeeze(-1)

  @torch.no_grad()
  def reward(self, transitions: torch.Tensor, eps: float = 1.0e-4) -> torch.Tensor:
    expert_like_prob = torch.sigmoid(self.forward(transitions))
    return -torch.log(torch.clamp(1.0 - expert_like_prob, min=eps))


class AmpPolicyBuffer:
  """Small independent replay buffer for policy AMP transitions."""

  def __init__(self, capacity: int, device: str | torch.device) -> None:
    self.capacity = int(capacity)
    self.device = torch.device(device)
    self._chunks: list[torch.Tensor] = []
    self._size = 0

  @property
  def size(self) -> int:
    return self._size

  def add(self, transitions: torch.Tensor) -> None:
    transitions = transitions.detach().to(self.device)
    if transitions.ndim != 2:
      raise ValueError(f"AMP policy transitions must be [N, D], got {tuple(transitions.shape)}")
    self._chunks.append(transitions)
    self._size += int(transitions.shape[0])
    self._trim()

  def _trim(self) -> None:
    while self._size > self.capacity and self._chunks:
      first = self._chunks[0]
      overflow = self._size - self.capacity
      if overflow >= first.shape[0]:
        self._chunks.pop(0)
        self._size -= int(first.shape[0])
      else:
        self._chunks[0] = first[overflow:]
        self._size -= int(overflow)

  def sample(self, batch_size: int) -> torch.Tensor:
    if self._size <= 0:
      raise ValueError("AMP policy buffer is empty")
    all_transitions = torch.cat(self._chunks, dim=0)
    idx = torch.randint(0, all_transitions.shape[0], (int(batch_size),), device=all_transitions.device)
    return all_transitions[idx]


def _resolve_manifest_path(demo_data_dir: str | None, manifest_path: str | None) -> Path | None:
  if manifest_path:
    return Path(manifest_path).expanduser()
  if demo_data_dir:
    return Path(demo_data_dir).expanduser() / "manifest.json"
  return None


def _get_mapping_value(obs: Any, key: str) -> Any:
  if isinstance(obs, Mapping):
    return obs[key]
  try:
    return obs[key]
  except Exception as exc:  # pragma: no cover - TensorDict raises version-specific errors
    raise KeyError(key) from exc


def _flatten_obs_group(value: torch.Tensor) -> torch.Tensor:
  if value.ndim == 1:
    return value.unsqueeze(0)
  if value.ndim > 2:
    return value.flatten(start_dim=1)
  return value


def _infer_obs_group_dim(obs: Any, group: str) -> int | None:
  try:
    value = _get_mapping_value(obs, group)
  except KeyError:
    return None
  if not torch.is_tensor(value):
    return None
  return int(_flatten_obs_group(value).shape[-1])


class GetupAmpPPO(PPO):
  """PPO with an AMP discriminator reward for G1 GetUp demos."""

  @staticmethod
  def construct_algorithm(obs: Any, env: Any, cfg: dict, device: str) -> "GetupAmpPPO":
    amp_group = cfg["algorithm"].get("amp_obs_group", "amp")
    cfg["algorithm"].setdefault("amp_obs_dim", _infer_obs_group_dim(obs, amp_group))
    if cfg["algorithm"].get("amp_obs_dim") is None:
      raise ValueError(f"AMP observation group {amp_group!r} is missing from env observations")
    # Forward env step_dt so the expert dataset can resample demos to match
    # the policy's transition spacing.  Falls back to 0.02 (50 Hz) when the
    # env wrapper does not expose step_dt (e.g., unit-test fakes).
    env_step_dt = getattr(getattr(env, "unwrapped", env), "step_dt", None)
    if env_step_dt is None:
      env_step_dt = getattr(env, "step_dt", 0.02)
    cfg["algorithm"].setdefault("amp_target_dt", float(env_step_dt))
    return PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[return-value]

  def __init__(
    self,
    *args,
    demo_data_dir: str | None = None,
    manifest_path: str | None = None,
    amp_reward_scale: float = 0.25,
    amp_obs_group: str = "amp",
    amp_obs_dim: int | None = None,
    discriminator_hidden_dims: tuple[int, ...] = (256, 128),
    discriminator_learning_rate: float = 1.0e-4,
    amp_batch_size: int = 256,
    amp_buffer_capacity: int = 65536,
    discriminator_grad_penalty: float = 0.0,
    require_demo_data: bool = True,
    amp_target_dt: float = 0.02,
    amp_getup_segments: bool = True,
    amp_feature_layout: str = "yaw_invariant",
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.demo_data_dir = demo_data_dir
    self.manifest_path = _resolve_manifest_path(demo_data_dir, manifest_path)
    self.amp_reward_scale = float(amp_reward_scale)
    self.amp_obs_group = amp_obs_group
    self.amp_batch_size = int(amp_batch_size)
    self.discriminator_grad_penalty = float(discriminator_grad_penalty)
    self.amp_policy_buffer = AmpPolicyBuffer(amp_buffer_capacity, self.device)
    self.last_amp_stats: dict[str, float] = {}

    self.expert_dataset: AmpExpertDataset | None = None
    if self.manifest_path is not None and self.manifest_path.exists():
      self.expert_dataset = AmpExpertDataset(
        self.manifest_path,
        device=self.device,
        target_dt=float(amp_target_dt),
        getup_segments=bool(amp_getup_segments),
        feature_layout=str(amp_feature_layout),
      )
      amp_obs_dim = self.expert_dataset.obs_dim
    elif require_demo_data:
      raise FileNotFoundError(
        "AMP demo manifest is required. Run scripts/prepare_g1_getup_amp_data.py first; "
        f"expected: {self.manifest_path}"
      )
    if amp_obs_dim is None:
      raise ValueError("amp_obs_dim could not be inferred; provide an amp observation group or manifest")

    self.amp_obs_dim = int(amp_obs_dim)
    self.discriminator = AmpDiscriminator(
      input_dim=2 * self.amp_obs_dim,
      hidden_dims=tuple(int(v) for v in discriminator_hidden_dims),
    ).to(self.device)
    self.amp_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=float(discriminator_learning_rate))

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def _amp_obs(self, obs: Any) -> torch.Tensor:
    amp_obs = _flatten_obs_group(_get_mapping_value(obs, self.amp_obs_group)).to(self.device)
    if amp_obs.shape[-1] != self.amp_obs_dim:
      raise ValueError(
        f"AMP obs dim mismatch: env produced {amp_obs.shape[-1]}, discriminator expects {self.amp_obs_dim}"
      )
    return amp_obs

  def _amp_transition(self, prev_obs: Any, next_obs: Any) -> torch.Tensor:
    return torch.cat([self._amp_obs(prev_obs), self._amp_obs(next_obs)], dim=-1)

  def process_env_step(
    self,
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
  ) -> None:
    prev_obs = self.transition.observations
    if prev_obs is None:
      super().process_env_step(obs, rewards, dones, extras)
      return

    transitions = self._amp_transition(prev_obs, obs)
    self.amp_policy_buffer.add(transitions)
    with torch.no_grad():
      amp_reward = self.discriminator.reward(transitions)
    reward_view = amp_reward.reshape_as(rewards) if rewards.shape == amp_reward.shape else amp_reward.view(-1, 1)
    augmented_rewards = rewards + self.amp_reward_scale * reward_view.to(rewards.device, dtype=rewards.dtype)

    extras.setdefault("log", {})
    extras["log"]["amp/reward_mean"] = amp_reward.mean().detach()
    extras["log"]["amp/policy_score"] = torch.sigmoid(self.discriminator(transitions)).mean().detach()
    super().process_env_step(obs, augmented_rewards, dones, extras)

  def _discriminator_loss(
    self,
    expert_transitions: torch.Tensor,
    policy_transitions: torch.Tensor,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    expert_logits = self.discriminator(expert_transitions)
    policy_logits = self.discriminator(policy_transitions)
    expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
    policy_loss = F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
    loss = expert_loss + policy_loss
    if self.discriminator_grad_penalty > 0.0:
      loss = loss + self.discriminator_grad_penalty * (
        expert_logits.square().mean() + policy_logits.square().mean()
      )
    with torch.no_grad():
      expert_score = torch.sigmoid(expert_logits).mean().item()
      policy_score = torch.sigmoid(policy_logits).mean().item()
      reward_mean = self.discriminator.reward(policy_transitions).mean().item()
    return loss, {
      "amp/expert_score": float(expert_score),
      "amp/policy_score": float(policy_score),
      "amp/reward_mean": float(reward_mean),
    }

  def update_amp_discriminator(self) -> dict[str, float]:
    if self.expert_dataset is None or self.amp_policy_buffer.size == 0:
      return {
        "amp/discriminator_loss": 0.0,
        "amp/expert_score": 0.0,
        "amp/policy_score": 0.0,
        "amp/reward_mean": 0.0,
      }
    batch_size = max(1, min(self.amp_batch_size, self.amp_policy_buffer.size))
    expert_transitions = self.expert_dataset.sample_transitions(batch_size)
    policy_transitions = self.amp_policy_buffer.sample(batch_size)
    loss, stats = self._discriminator_loss(expert_transitions, policy_transitions)
    self.amp_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
    self.amp_optimizer.step()
    stats["amp/discriminator_loss"] = float(loss.detach().item())
    self.last_amp_stats = stats
    return stats

  def update(self) -> dict[str, float]:
    amp_stats = self.update_amp_discriminator()
    ppo_stats = super().update()
    ppo_stats.update(amp_stats)
    return ppo_stats

  def save(self) -> dict:
    saved = super().save()
    saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
    saved["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
    saved["amp_config"] = {
      "demo_data_dir": self.demo_data_dir,
      "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
      "amp_reward_scale": self.amp_reward_scale,
      "amp_obs_group": self.amp_obs_group,
      "amp_obs_dim": self.amp_obs_dim,
    }
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("amp", True):
      if "amp_discriminator_state_dict" in loaded_dict:
        self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
      if "amp_optimizer_state_dict" in loaded_dict:
        self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
    return load_iteration
