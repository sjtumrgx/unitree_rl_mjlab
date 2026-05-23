from __future__ import annotations

import types

import pytest

from src.tasks.velocity.rl import antifall_deploy_contract as deploy


def test_antifall_getup_extra_actor_terms_have_explicit_deploy_dims() -> None:
  assert deploy._actor_term_dim("getup_progress", joint_dim=29) == 5
  assert deploy._actor_term_dim("bfm_local_body_state", joint_dim=29) == 448
  assert deploy._actor_term_dim("height_scan", joint_dim=29) == 187


def test_antifall_deploy_term_dims_match_train_projection_layout() -> None:
  from scripts.train import _g1_antifall_getup_actor_layout

  for term in _g1_antifall_getup_actor_layout():
    assert deploy._actor_term_dim(term.name, joint_dim=29) == len(term.feature_names)


def test_antifall_deploy_uses_per_term_actor_history_when_group_history_is_disabled() -> None:
  env = types.SimpleNamespace(
    cfg=types.SimpleNamespace(
      observations={
        "actor": types.SimpleNamespace(
          history_length=None,
          terms={
            "base_ang_vel": types.SimpleNamespace(history_length=6),
            "bfm_local_body_state": types.SimpleNamespace(history_length=0),
          },
        )
      }
    )
  )

  assert deploy._actor_term_history_length(env, "base_ang_vel") == 6
  assert deploy._actor_term_history_length(env, "bfm_local_body_state") == 1


def test_antifall_deploy_contract_still_rejects_unknown_actor_terms() -> None:
  with pytest.raises(KeyError, match="Unsupported anti-fall actor term"):
    deploy._actor_term_dim("unknown_term", joint_dim=29)
