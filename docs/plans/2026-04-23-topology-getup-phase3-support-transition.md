# Topology Get-Up Phase-3 Support-Transition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove a reward conflict that keeps topology-getup policies on hand/knee supports instead of transitioning toward standing.

**Architecture:** The bounded phase-2 run showed support-body contact count rising while torso clearance and body orientation worsened, and the support-contact diversity reward increased with it. Implement a single focused fix: gate support-contact diversity reward to the early low-torso phase, then add a late support-body contact penalty once the torso has lifted enough that non-foot contacts should start disappearing.

**Tech Stack:** Python, pytest, torch, existing topology-getup reward helpers

---

### Task 1: Add failing tests for support-transition shaping

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`
- Reference: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Reference: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Write a failing test for early-only support diversity reward**
- Mock support contact count = 2.
- Assert the reward is positive when torso height is low.
- Assert it becomes zero once torso height is above the activation ceiling.

**Step 2: Write a failing test for late support-contact penalty**
- Mock support contact count > 0.
- Assert the penalty is zero while torso is low.
- Assert the penalty becomes positive after torso lift.

**Step 3: Extend contract assertions**
- Assert topology-getup reward config still contains support-contact diversity reward.
- Assert it now uses the early-phase gating params.
- Assert the late support-contact penalty reward term exists.

**Step 4: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL because the new reward behavior/config is not wired yet.

### Task 2: Implement the minimal support-transition fix

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Gate support-contact diversity to the early phase**
- Add an optional torso-height ceiling to `support_contact_diversity_reward(...)`.
- Use it from topology-getup config so the reward only applies before meaningful lift.

**Step 2: Add late support-contact penalty**
- Implement `support_body_contact_penalty_after_lift(...)` using normalized non-foot support count.
- Activate it only after torso height exceeds the lift threshold.
- Add it to topology-getup rewards with a conservative negative weight.

**Step 3: Run GREEN**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: PASS.

### Task 3: Re-run bounded verification

**Files:**
- Existing files only

**Step 1: Focused regression suite**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py tests/tasks/test_g1_topology_getup_registration.py tests/tasks/test_g1_topology_getup_naive_baseline_contract.py tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: PASS.

**Step 2: Fresh bounded naive run**
Run a short CPU naive-depth training pass and inspect:
- `support_body_contact_count`
- `support_contact_diversity_reward`
- `support_body_contact_penalty_after_lift`
- `torso_clearance`
- `getup_torso_lift_reward`
- `Train/mean_reward`

**Step 3: Summarize whether the reward conflict reduced**
- Report whether support contacts drop or stop being rewarded late.
- Report what still blocks actual stand-up success.
