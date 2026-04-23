# Topology Get-Up Phase-10 Stall-Termination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase reset frequency for hopeless low-progress getup episodes so PPO gets denser learning feedback instead of spending long horizons stuck near the floor.

**Architecture:** Add a topology-getup-specific class-based termination term that tracks the best get-up progress reached in the episode and terminates once a short grace period passes without enough lift/orientation progress. Keep the change local to topology-getup terminations/config and re-run bounded verification.

**Tech Stack:** Python, pytest, torch

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Extend contract expectations**
- Assert topology-getup config contains `stalled_getup` termination.

**Step 2: Write a failing unit test for the stall termination**
- Assert it does not fire before the grace step count.
- Assert it fires after the grace period when progress never crosses the threshold.
- Assert sufficient progress suppresses the termination.
- Assert `reset()` re-arms it.

**Step 3: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement the minimal stall termination

**Files:**
- Add/Modify: `src/tasks/velocity/mdp/topology_getup/terminations.py`
- Modify: `src/tasks/velocity/mdp/topology_getup/__init__.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add `stalled_getup_progress` class**
- Track best progress per env.
- Use torso-height progress and facing-up alignment.
- Terminate after a short grace window if best progress stays below threshold.

**Step 2: Wire the term into topology-getup config**
- Add conservative defaults so it only trims obviously hopeless episodes.

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
Inspect whether:
- `getup_upright` becomes nonzero, or
- `head_contact` / `time_out` distribution changes, or
- reward curves stop spending many iterations at zero-progess summaries.
