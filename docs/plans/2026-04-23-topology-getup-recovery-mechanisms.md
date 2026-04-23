# Topology Get-Up Recovery Mechanisms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve topology-getup recovery success by porting the most compatible ideas from HoST, PHC, and KungFuAthleteBot into the existing `topology_getup` lane.

**Architecture:** Port three narrow mechanisms that fit the current codebase: (1) a HoST-style dense height-progress reward so the policy gets shaped signal before full stand-up, (2) a PHC-style explicit recovery-episode marker so fallen-pose resets activate existing anti-fall recovery rewards/metrics, and (3) a KungFuAthleteBot-style tolerant head-contact termination so brief early recovery contacts do not instantly kill promising episodes. Keep the diff local to `src/tasks/velocity/mdp/topology_getup`, `src/tasks/velocity/mdp/terminations.py`, and the topology-getup env config.

**Tech Stack:** Python, pytest, torch, existing mjlab manager term patterns

---

### Task 1: Lock the intended ported behaviors with failing tests

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`
- Reference: `src/tasks/velocity/mdp/topology_getup/events.py`
- Reference: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Reference: `src/tasks/velocity/mdp/terminations.py`

**Step 1: Write a failing test for dense height-progress shaping**
- Assert a face-up torso at higher height gets more `getup_height_progress_reward(...)` than a low torso.
- Assert an upside-down pose gets zero progress reward.

**Step 2: Write a failing test for PHC-style recovery episode marking**
- Call `reset_root_state_from_presets(...)` with a mocked env and monkeypatched `reset_root_state_uniform`.
- Assert the anti-fall state records `disturbance_count == 1`, kind `DISTURBANCE_NEAR_FAILURE_RESET`, and an active recovery window immediately after reset.

**Step 3: Write a failing test for tolerant contact termination**
- Instantiate the new tolerant contact termination term with a mocked contact sensor.
- Assert brief contact inside the grace window does not terminate.
- Assert persistent contact beyond the threshold does terminate.

**Step 4: Run RED**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: FAIL because the new reward, recovery marker behavior, and tolerant termination are not implemented yet.

### Task 2: Implement the recovery mechanisms with minimal local edits

**Files:**
- Modify: `src/tasks/velocity/mdp/topology_getup/rewards.py`
- Modify: `src/tasks/velocity/mdp/topology_getup/events.py`
- Modify: `src/tasks/velocity/mdp/terminations.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`
- Modify: `src/tasks/velocity/mdp/topology_getup/__init__.py` only if needed for exports

**Step 1: Add HoST-style dense progress reward**
- Implement `getup_height_progress_reward(...)` as a monotonic torso-height progress reward gated by upright-facing alignment.
- Add it to the topology-getup reward config with a conservative weight.

**Step 2: Add PHC-style recovery episode marker at fallen reset**
- Update `reset_root_state_from_presets(...)` so each fallen-pose reset marks the env as a synthetic near-failure disturbance instead of clearing anti-fall state to all zeros.
- Keep the disturbance metadata minimal and deterministic enough for tests.

**Step 3: Add KungFuAthleteBot-style tolerant head-contact termination**
- Implement a class-based termination term that wraps illegal contact with a grace period and a persistence threshold.
- Replace the topology-getup `head_contact` termination with the tolerant variant.

**Step 4: Run GREEN**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`
Expected: PASS.

### Task 3: Protect topology-getup contracts and summarize what changed

**Files:**
- Modify if needed: `tests/tasks/test_g1_topology_getup_contracts.py`
- Existing files only otherwise

**Step 1: Extend contract coverage**
- Assert the new reward is present.
- Assert the head-contact termination now points to the tolerant term.

**Step 2: Run focused regression coverage**
Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py tests/tasks/test_g1_topology_getup_registration.py tests/tasks/test_g1_topology_getup_naive_baseline_contract.py tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
Expected: PASS.

**Step 3: Run a lightweight syntax/compile check**
Run: `python -m compileall src/tasks/velocity/mdp/topology_getup src/tasks/velocity/mdp/terminations.py tests/tasks/test_g1_topology_getup_contracts.py`
Expected: PASS.

**Step 4: Record residual follow-up**
- Note that full PHC-style scheduled recovery episode mixing and HoST-style external pull-force are intentionally deferred.
- Note that retraining is required to measure real success-rate improvement.
