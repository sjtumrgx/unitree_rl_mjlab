from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from src.tasks.velocity.rl.getup_amp import AmpDiscriminator, GetupAmpPPO
from src.tasks.velocity.rl.getup_amp_data import AmpExpertDataset, prepare_amp_dataset

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "g1_getup_amp"


class _FakeNet(nn.Module):
  is_recurrent = False

  def __init__(self, output_dim: int):
    super().__init__()
    self.bias = nn.Parameter(torch.zeros(output_dim))
    self.output_distribution_params = (torch.zeros(1, output_dim), torch.ones(1, output_dim))
    self.output_entropy = torch.ones(1)
    self.output_std = torch.ones(output_dim)
    self.output_dim = output_dim

  def forward(self, obs, *args, **kwargs):
    batch = obs["actor"].shape[0] if isinstance(obs, dict) and "actor" in obs else obs["amp"].shape[0]
    out = self.bias.unsqueeze(0).repeat(batch, 1)
    self.output_distribution_params = (out.detach(), torch.ones_like(out))
    self.output_entropy = torch.ones(batch, device=out.device)
    return out

  def update_normalization(self, obs) -> None:
    del obs

  def reset(self, dones) -> None:
    del dones

  def get_hidden_state(self):
    return None

  def get_output_log_prob(self, actions):
    return torch.zeros(actions.shape[0], device=actions.device)

  def get_kl_divergence(self, old_distribution_params, distribution_params):
    del old_distribution_params, distribution_params
    return torch.zeros(1)


class _FakeStorage:
  def __init__(self):
    self.rewards: list[torch.Tensor] = []
    self.cleared = False

  def add_transition(self, transition) -> None:
    self.rewards.append(transition.rewards.clone())

  def clear(self) -> None:
    self.cleared = True

  def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
    del num_mini_batches, num_learning_epochs
    return iter(())

  def recurrent_mini_batch_generator(self, num_mini_batches, num_learning_epochs):
    del num_mini_batches, num_learning_epochs
    return iter(())


def _prepared_manifest(tmp_path: Path) -> Path:
  out = tmp_path / "prepared"
  prepare_amp_dataset(_FIXTURE_DIR, out, validate_only=True)
  return out / "manifest.json"


def _make_algorithm(tmp_path: Path) -> tuple[GetupAmpPPO, _FakeStorage]:
  manifest = _prepared_manifest(tmp_path)
  storage = _FakeStorage()
  alg = GetupAmpPPO(
    _FakeNet(2),
    _FakeNet(1),
    storage,
    device="cpu",
    manifest_path=str(manifest),
    amp_reward_scale=0.5,
    amp_obs_group="amp",
    amp_batch_size=4,
    amp_buffer_capacity=16,
    num_learning_epochs=1,
    num_mini_batches=1,
  )
  return alg, storage


def test_amp_expert_dataset_and_discriminator_are_finite(tmp_path: Path) -> None:
  dataset = AmpExpertDataset(_prepared_manifest(tmp_path))
  transitions = dataset.sample_transitions(4)
  discriminator = AmpDiscriminator(transitions.shape[-1], hidden_dims=(16,))
  logits = discriminator(transitions)
  reward = discriminator.reward(transitions)

  assert logits.shape == (4,)
  assert torch.isfinite(logits).all()
  assert torch.isfinite(reward).all()


def test_process_env_step_adds_amp_reward_before_storage(tmp_path: Path) -> None:
  alg, storage = _make_algorithm(tmp_path)
  prev_amp = alg.expert_dataset.sample_observations(3)  # type: ignore[union-attr]
  next_amp = prev_amp + 0.01
  rewards = torch.zeros(3)
  dones = torch.zeros(3, dtype=torch.bool)
  extras: dict[str, object] = {}

  alg.transition.observations = {"amp": prev_amp, "actor": prev_amp[:, :2]}
  alg.transition.values = torch.zeros(3, 1)
  alg.process_env_step({"amp": next_amp, "actor": next_amp[:, :2]}, rewards, dones, extras)

  assert storage.rewards
  assert torch.all(storage.rewards[0] > rewards)
  assert alg.amp_policy_buffer.size == 3
  assert "amp/reward_mean" in extras["log"]  # type: ignore[index]


def test_update_returns_finite_amp_loss_and_buffer_survives_storage_clear(tmp_path: Path) -> None:
  alg, storage = _make_algorithm(tmp_path)
  prev_amp = alg.expert_dataset.sample_observations(3)  # type: ignore[union-attr]
  next_amp = prev_amp + 0.01
  alg.transition.observations = {"amp": prev_amp, "actor": prev_amp[:, :2]}
  alg.transition.values = torch.zeros(3, 1)
  alg.process_env_step({"amp": next_amp, "actor": next_amp[:, :2]}, torch.zeros(3), torch.zeros(3, dtype=torch.bool), {})

  loss = alg.update()

  assert storage.cleared is True
  assert alg.amp_policy_buffer.size == 3
  assert torch.isfinite(torch.tensor(loss["amp/discriminator_loss"]))
  assert torch.isfinite(torch.tensor(loss["amp/reward_mean"]))
  assert loss["value"] == 0
  assert loss["surrogate"] == 0


def test_amp_save_load_preserves_discriminator_state(tmp_path: Path) -> None:
  alg, _ = _make_algorithm(tmp_path)
  with torch.no_grad():
    for param in alg.discriminator.parameters():
      param.fill_(0.123)
  saved = alg.save()

  restored, _ = _make_algorithm(tmp_path)
  restored.load(saved, load_cfg={"actor": False, "critic": False, "optimizer": False, "iteration": False, "amp": True}, strict=True)

  for left, right in zip(alg.discriminator.parameters(), restored.discriminator.parameters()):
    assert torch.allclose(left, right)


def test_amp_expert_dataset_rejects_stop_source_gate(tmp_path: Path) -> None:
  manifest = _prepared_manifest(tmp_path)
  (manifest.parent / "source_gate.json").write_text(
    '{"status": "STOP", "stop_reasons": ["license unresolved"]}'
  )

  import pytest

  with pytest.raises(ValueError, match="blocks training"):
    AmpExpertDataset(manifest)
