# Topology Get-Up Phase-4 Orientation/Transition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Shift topology-getup shaping out of the prolonged low-support regime by rewarding intermediate facing-up progress and moving the support-transition gate earlier.

**Architecture:** The phase-3 run shows support-body contact count still rises while torso clearance drops and support-contact diversity reward remains active late. Add a simple dense facing-up reward, then move the support-transition thresholds earlier so the low-support reward turns off and the late support penalty turns on sooner.

**Tech Stack:** Python, pytest, torch, topology-getup reward helpers

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Write failing test for facing-up reward**
- Assert it is larger for a face-up body than for an upside-down body.

**Step 2: Tighten contract expectations**
- Assert topology-getup config contains `getup_facing_up_reward`.
- Assert support-contact diversity gate is now `0.2`.
- Assert support penalty activation height is now `0.2`.

**Step 3: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement the minimal tuning change

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add dense facing-up reward**
- Implement `getup_facing_up_reward(...) = clamp(upright_alignment, 0, 1)`.
- Add it to topology-getup rewards with a moderate positive weight.

**Step 2: Shift support-transition thresholds earlier**
- Change `support_contact_diversity_reward.active_below_height` to `0.2`.
- Change `support_body_contact_penalty_after_lift.activation_height` to `0.2`.

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
Run a short CPU `naive_depth` training pass and compare:
- `getup_facing_up_reward`
- `support_contact_diversity_reward`
- `support_body_contact_penalty_after_lift`
- `support_body_contact_count`
- `torso_clearance`
- `Train/mean_reward`

**Step 3: Summarize whether the policy exits the low-support phase earlier**
