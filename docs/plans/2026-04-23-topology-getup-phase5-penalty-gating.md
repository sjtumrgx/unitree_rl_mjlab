# Topology Get-Up Phase-5 Penalty-Gating Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce an evidence-backed reward conflict where large early smoothness penalties appear to suppress the aggressive motion needed to transition out of the fallen state.

**Architecture:** The bounded phase-4 run still ends with zero get-up success while `action_rate_l2` remains one of the largest negative terms. Port a narrow HoST-inspired relaxation: replace the inherited `action_rate_l2` term with a topology-getup-specific wrapper that only activates after the torso has lifted past the early recovery phase.

**Tech Stack:** Python, pytest, torch, topology-getup reward helpers

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Write a failing test for gated action-rate penalty**
- Assert the wrapper returns zero before torso lift.
- Assert it becomes positive after torso lift for a nonzero action delta.

**Step 2: Extend contract expectations**
- Assert topology-getup config still includes `action_rate_l2` but now points to the gated wrapper.

**Step 3: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement minimal penalty gating

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add `action_rate_after_lift(...)`**
- Wrap the base `action_rate_l2` reward and gate it by torso height.

**Step 2: Rebind topology-getup `action_rate_l2`**
- Override the inherited reward term in topology-getup config to use the gated wrapper.

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
Run a short CPU naive-depth training pass and inspect whether the negative `action_rate_l2` term stays near zero longer while torso-lift rewards remain active.

**Step 3: Summarize whether early motion suppression reduced**
