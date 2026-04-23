# Topology Get-Up Phase-9 Progress-Observation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit compact get-up progress observation so the policy no longer has to infer lift/orientation/support-transition state only from delayed rewards.

**Architecture:** Add a topology-getup-specific observation term `getup_progress` containing compact scalar progress features (torso-height progress, facing-up alignment, normalized support-contact count). Wire it into the actor and critic observation groups, then re-run bounded training verification.

**Tech Stack:** Python, pytest, torch

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Extend actor/critic contract expectations**
- Assert actor terms now include `getup_progress`.
- Assert critic terms also include `getup_progress`.

**Step 2: Write a failing unit test for the observation helper**
- Mock torso height, projected gravity, and support-contact count.
- Assert the output grows with height/alignment and reflects support-contact count.

**Step 3: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement the minimal observation change

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/observations.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add `getup_progress_features(...)`**
- Return a compact vector of progress features.

**Step 2: Wire it into actor and critic groups**
- Keep the term local to topology-getup.

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
Run a short CPU `naive_depth` pass and inspect whether:
- `getup_upright` becomes nonzero, or
- short-horizon shaping metrics improve further.
