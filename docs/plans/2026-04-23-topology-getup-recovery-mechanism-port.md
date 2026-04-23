# Topology Get-Up Recovery Mechanism Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase `topology_getup` recovery success by porting the most compatible recovery mechanisms from HoST, PHC, and KungFuAthleteBot into the current Unitree task.

**Architecture:** Keep the solution reward- and reset-driven inside the existing `topology_getup` task family. Port the highest-value ideas that fit the current codebase: (1) PHC-style physically plausible fallen-state recovery starts, (2) HoST-style staged recovery shaping around support/contact transitions, and (3) KungFuAthleteBot-style easy-to-hard curriculum emphasis on robustness. Start with the highest-confidence blocker first: current fallen resets are relative to the standing default root state, so the robot starts airborne instead of already fallen.

**Tech Stack:** Python, mjlab manager-based env config, MuJoCo/Warp simulation, pytest, TensorBoard scalar inspection.

---

### Task 1: Lock the reset contract that should produce true fallen starts

**Files:**
- Modify: `tests/tasks/test_g1_topology_getup_contracts.py`
- Modify: `src/tasks/velocity/config/g1_topology_getup/env_cfgs.py`

**Step 1: Write the failing test**

Add a contract test asserting the Stage0 fallen presets lower the base from the standing default frame by using negative `z` offsets (rather than `0..0.05`, which leaves the root near standing height because resets are relative to `default_root_state`).

**Step 2: Run test to verify it fails**

Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`

Expected: FAIL because current preset `z` ranges are non-negative.

**Step 3: Write minimal implementation**

Change the preset `z` offsets in `env_cfgs.py` so supine/side starts are near terrain contact and seated starts are slightly higher but still clearly below standing height.

**Step 4: Run test to verify it passes**

Run: `pytest tests/tasks/test_g1_topology_getup_contracts.py -q`

Expected: PASS.

### Task 2: Verify the reset fix produces contact-rich fallen states

**Files:**
- No production-file changes required if Task 1 is sufficient
- Evidence script only (one-off inline Python)

**Step 1: Write the failing diagnostic expectation**

Use a direct environment probe to show the current bug: default standing-relative resets make `support_body_contact_count == 0`, `feet_contact == 0`, and misleadingly large body clearance immediately after reset.

**Step 2: Run probe before/after**

Run a short inline Python probe that instantiates `Unitree-G1-TopologyGetUp-Stage0-NaiveDepth`, resets 8 CPU envs, and prints support contacts / feet contacts / torso clearance.

Expected after the fix: non-zero body support on reset or within the first step, with torso/pelvis clearance close to terrain rather than standing-height-like values.

### Task 3: Re-run focused regression tests

**Files:**
- Existing test files only

**Step 1: Run targeted tests**

Run:
- `pytest tests/tasks/test_g1_topology_getup_contracts.py tests/tasks/test_g1_topology_getup_upright_metrics.py -q`
- `pytest tests/tasks/test_g1_topology_getup_export_artifacts.py tests/scripts/test_promote_topology_getup_artifact.py -q`

Expected: PASS.

### Task 4: Run a fresh bounded training probe and inspect learning signals

**Files:**
- No code changes unless new blocker appears

**Step 1: Launch bounded train**

Run:
`python scripts/train.py Unitree-G1-TopologyGetUp-Stage0-NaiveDepth --agent.max_iterations=10 --agent.logger=tensorboard --env.scene.num-envs=8 --gpu-ids cpu`

**Step 2: Inspect TensorBoard event scalars**

Check:
- `Episode_Metrics/getup_upright`
- `Episode_Metrics/getup_success_count`
- `Episode_Metrics/support_body_contact_count`
- `Episode_Reward/getup_facing_up_reward`
- `Episode_Reward/recovery_quality`

Expected: improved early recovery signal quality versus the airborne-reset baseline; if success is still zero, use the new evidence to decide whether the next port should be HoST-like foot-support shaping or PHC-like fall-state bank generation.
