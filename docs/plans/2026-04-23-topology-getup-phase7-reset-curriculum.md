# Topology Get-Up Phase-7 Reset-Curriculum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the earliest topology-getup training distribution easier so the policy can learn a solvable stand-up basin before seeing the full fallen-pose mix.

**Architecture:** Add a reset-preset curriculum to `reset_root_state_from_presets(...)` driven by `env.common_step_counter`. Start with seated-fall only, then mix in side-lying, then finally enable the full uniform mix including supine. Keep the change local to topology-getup events/config and lock it with deterministic tests.

**Tech Stack:** Python, pytest, torch, existing curriculum-stage pattern

---

### Task 1: Add failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`

**Step 1: Write a failing test for preset-weight schedule resolution**
- Assert the helper returns stage-appropriate weights for early, mid, and late `common_step_counter` values.

**Step 2: Write a failing test for weighted preset sampling path**
- Monkeypatch `torch.multinomial` to capture the weight vector passed into `reset_root_state_from_presets(...)`.
- Assert the configured stage weights are used instead of uniform sampling.

**Step 3: Extend contract expectations**
- Assert topology-getup reset config contains the preset curriculum stage list.

**Step 4: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL.

### Task 2: Implement the minimal reset curriculum

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/events.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Add a helper to resolve preset weights from step stages**
- Similar to existing reward/command curriculum helpers, use `env.common_step_counter`.

**Step 2: Teach `reset_root_state_from_presets(...)` to sample with stage weights**
- When curriculum stages are present, choose the current weight vector and call `torch.multinomial`.
- Fall back to uniform behavior otherwise.

**Step 3: Wire a conservative curriculum**
- Early: seated-only
- Middle: seated + side-lying
- Late: full mix including supine

**Step 4: Run GREEN**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: PASS.

### Task 3: Re-run bounded verification

**Files:**
- Existing files only

**Step 1: Focused regression suite**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py tests/tasks/test_g1_topology_getup_registration.py tests/tasks/test_g1_topology_getup_naive_baseline_contract.py tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: PASS.

**Step 2: Fresh bounded naive run**
Run a short CPU naive-depth pass and inspect whether the early iterations show stronger facing-up / lift / height-progress signal and whether `getup_upright` can become nonzero.

**Step 3: Summarize whether the easier reset curriculum materially improves short-horizon learnability**
