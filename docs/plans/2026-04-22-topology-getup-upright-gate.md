# Topology Get-Up Upright Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent topology get-up training from counting upside-down supine resets as successful upright recoveries.

**Architecture:** Add an explicit upright-facing gate to the topology get-up success metric and posture reward so upside-down states no longer satisfy the success/bonus logic. Lock the bug with small unit tests that exercise the pure reward/metric helpers using mocked env state instead of a full simulator.

**Tech Stack:** Python, pytest, mjlab task helpers, torch

---

### Task 1: Lock the upside-down regression with focused tests

**Files:**
- Create/Modify: `tests/tasks/test_g1_topology_getup_upright_metrics.py`
- Reference: `src/tasks/velocity/mdp/topology_getup/metrics.py`
- Reference: `src/tasks/velocity/mdp/topology_getup/rewards.py`

**Step 1: Write failing tests**
- Add a mocked env/robot state where `projected_gravity_b[:, :2] == 0`, torso height is above threshold, but `projected_gravity_b[:, 2] > 0` (upside-down).
- Assert `getup_upright(...) == 0` for that state.
- Assert an actually upright state with `projected_gravity_b[:, 2] < 0` still passes.
- Assert `getup_posture_reward(...)` scores the upright state higher than the upside-down state.

**Step 2: Run the targeted test to verify RED**
Run: `pytest tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: FAIL because upside-down states currently count as upright / receive high get-up posture reward.

### Task 2: Fix the upright gate in topology get-up helpers

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/metrics.py`
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`

**Step 1: Implement minimal code**
- Add a shared upright-facing check based on `projected_gravity_b[:, 2]` so only torso-up orientations satisfy get-up success.
- Reuse the same gate in `getup_posture_reward(...)` so upside-down supine poses do not get near-max posture reward.
- Keep the change local to topology-getup helpers; do not widen scope into unrelated anti-fall tasks in this pass.

**Step 2: Run the targeted tests to verify GREEN**
Run: `pytest tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: PASS.

### Task 3: Verify no topology-getup contracts regress

**Files:**
- Existing tests only

**Step 1: Run focused regression coverage**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py tests/tasks/test_g1_topology_getup_registration.py tests/tasks/test_g1_topology_getup_naive_baseline_contract.py tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: PASS.

**Step 2: Re-run a lightweight env probe**
- Instantiate the topology-getup env on CPU.
- Reset a small batch and confirm `getup_upright` at reset is no longer dominated by supine presets.

**Step 3: Summarize impact**
- Note that this fixes a reward/metric bug first.
- Reassess whether demo/distillation is still needed only after retraining with corrected success logic.
