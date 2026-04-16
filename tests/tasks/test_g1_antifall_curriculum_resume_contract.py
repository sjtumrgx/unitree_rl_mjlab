import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from src.tasks.velocity.rl.curriculum_state import stage_observation_signature, transition_load_mode


BOUNDARIES = (
  ("Unitree-G1-AntiFall-Stage0", "Unitree-G1-AntiFall-Stage1", "full"),
  ("Unitree-G1-AntiFall-Stage1", "Unitree-G1-AntiFall-Stage2", "full"),
  ("Unitree-G1-AntiFall-Stage2", "Unitree-G1-AntiFall-Stage3", "actor_only"),
  ("Unitree-G1-AntiFall-Stage3", "Unitree-G1-AntiFall-Stage4a", "actor_only"),
  ("Unitree-G1-AntiFall-Stage4a", "Unitree-G1-AntiFall-Stage4b", "full"),
)


def test_adjacent_stage_transitions_choose_expected_load_modes() -> None:
  for source_task, target_task, expected_mode in BOUNDARIES:
    assert transition_load_mode(source_task, target_task) == expected_mode


def test_actor_signature_stays_fixed_while_stage3_critic_signature_changes() -> None:
  stage2 = stage_observation_signature("Unitree-G1-AntiFall-Stage2")
  stage3 = stage_observation_signature("Unitree-G1-AntiFall-Stage3")
  stage4a = stage_observation_signature("Unitree-G1-AntiFall-Stage4a")

  assert stage2.actor == stage3.actor == stage4a.actor
  assert stage2.critic != stage3.critic
  assert stage3.critic != stage4a.critic
