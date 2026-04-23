# Topology Get-Up Phase-2 Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen topology-getup learning signal after the first recovery-mechanism port proved structurally correct but still failed to produce early stand-up success in bounded training.

**Architecture:** Port more of HoST’s stage-gated shaping into the current topology-getup lane: add an ungated torso-lift reward for early recovery, add one-shot torso-height stage bonuses, and gate the inherited stand-still penalty so it only activates after the robot has meaningfully lifted and oriented toward an upright pose. Keep the change local to topology-getup rewards/config plus regression tests.

**Tech Stack:** Python, pytest, torch, existing mjlab reward-term patterns

---

### Task 1: Add failing tests for phase-2 shaping

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`
- Reference: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Reference: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Write failing test for torso-lift reward**
- Assert `getup_torso_lift_reward(...)` grows with torso height even before full upright success.

**Step 2: Write failing test for one-shot stage bonus**
- Instantiate the stage-bonus reward term with a mocked env.
- Assert each threshold crossing pays once.
- Assert repeated calls at the same height do not re-pay.
- Assert `reset()` re-arms the bonus.

**Step 3: Write failing test for gated stand-still penalty**
- Assert the wrapped stand-still penalty is zero while torso height is low.
- Assert it becomes positive once torso height is high and the body is facing up.

**Step 4: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL because the new reward helpers and config wiring do not exist yet.

### Task 2: Implement the minimal phase-2 shaping changes

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add ungated torso-lift reward**
- Reward torso-height progress directly, without requiring full upright orientation.

**Step 2: Add HoST-style stage bonus**
- Pay one-shot bonuses at 2–3 torso-height milestones.
- Keep it resettable and deterministic.

**Step 3: Gate stand-still as a post-getup penalty**
- Override the inherited `stand_still` term with a topology-getup-specific wrapper that only applies after the robot has lifted enough and is facing up.

**Step 4: Run GREEN**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: PASS.

### Task 3: Run focused verification and bounded training evidence

**Files:**
- Existing files only

**Step 1: Regression suite**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py tests/tasks/test_g1_topology_getup_registration.py tests/tasks/test_g1_topology_getup_naive_baseline_contract.py tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: PASS.

**Step 2: Bounded naive training probe**
Run a short CPU naive-depth run and inspect whether the new torso-lift / stage-bonus rewards activate.

**Step 3: Summarize whether phase-2 shaping improved early learning signals**
- Report what changed in the logged rewards.
- Report what still requires full training.
