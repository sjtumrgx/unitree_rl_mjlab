# Topology Get-Up Phase-8 Gated Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove a likely inherited conflict where zero-command velocity tracking rewards encourage staying still before the robot has actually stood up.

**Architecture:** Override topology-getup's inherited `track_linear_velocity` and `track_angular_velocity` terms with wrappers that only activate after the torso has lifted and the body is facing sufficiently upward. This keeps post-stand stabilization rewards but stops them from suppressing the get-up motion itself.

**Tech Stack:** Python, pytest, torch, existing reward wrappers

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Write a failing test for gated tracking rewards**
- Assert both wrappers return zero before the activation gate.
- Assert they become positive once the torso is high enough and facing up.

**Step 2: Extend contract assertions**
- Assert topology-getup config points `track_linear_velocity` and `track_angular_velocity` to the gated wrappers.

**Step 3: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement the minimal gating fix

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add gated tracking wrappers**
- Reuse the existing base tracking rewards.
- Gate them by torso height + upright alignment threshold.

**Step 2: Rebind topology-getup reward config**
- Override inherited tracking terms to use the new wrappers.

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
Inspect whether early training keeps stronger lift/orientation rewards while reducing the bias toward low-motion fallen states.
