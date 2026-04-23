# Topology Get-Up Phase-6 Orientation Bonus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Strengthen the rotation-to-upright signal, since the assist-force branch improved lift/support metrics but `getup_upright` still stayed zero while facing-up reward remained tiny.

**Architecture:** Add a one-shot orientation phase bonus based on upright alignment thresholds and raise the continuous facing-up reward weight. Keep the change local to topology-getup rewards/config plus regression tests.

**Tech Stack:** Python, pytest, torch

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Write a failing test for orientation phase bonus**
- Instantiate the bonus term with mocked projected-gravity states.
- Assert each alignment threshold bonus pays once.
- Assert reset re-arms it.

**Step 2: Extend contract expectations**
- Assert `getup_orientation_phase_bonus` exists in topology-getup rewards.

**Step 3: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement the minimal orientation-bonus change

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add `getup_orientation_phase_bonus`**
- Use `_upright_alignment` thresholds and one-shot stage payments.

**Step 2: Wire the reward**
- Add the bonus term to topology-getup rewards.
- Increase `getup_facing_up_reward` weight modestly.

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
Run a short CPU naive-depth pass and inspect:
- `getup_facing_up_reward`
- `getup_orientation_phase_bonus`
- `getup_upright`
- `getup_success_count`
- `Train/mean_reward`

**Step 3: Summarize whether orientation progress becomes materially stronger**
