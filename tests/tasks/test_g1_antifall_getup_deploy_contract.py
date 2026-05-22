from __future__ import annotations

import pytest

from src.tasks.velocity.rl import antifall_deploy_contract as deploy


def test_antifall_getup_extra_actor_terms_have_explicit_deploy_dims() -> None:
  assert deploy._actor_term_dim("getup_progress", joint_dim=29) == 8
  assert deploy._actor_term_dim("bfm_local_body_state", joint_dim=29) == 433


def test_antifall_deploy_contract_still_rejects_unknown_actor_terms() -> None:
  with pytest.raises(KeyError, match="Unsupported anti-fall actor term"):
    deploy._actor_term_dim("unknown_term", joint_dim=29)
