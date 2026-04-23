# G1 AntiFall Deploy Target Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated `deploy/robots/g1_antifall` controller target plus export/config support so G1 anti-fall ONNX policies can be deployed without hand-editing the existing velocity controller.

**Architecture:** Reuse the existing G1 deploy runtime and RL state logic, but provide a separate robot folder and config that points to an anti-fall policy directory and uses an anti-fall-specific deploy YAML contract. Add a Python-side anti-fall deploy contract writer so exported anti-fall policies include the matching `deploy.yaml`, then keep curriculum top-level exports in sync by copying that deploy YAML alongside `policy.onnx`.

**Tech Stack:** C++17 deploy controller, YAML config, ONNX Runtime, existing deploy FSM/runtime headers, pytest.

---

### Task 1: Lock the anti-fall deploy contract with failing tests

**Files:**
- Create: `tests/scripts/test_g1_antifall_deploy_config.py`
- Modify: `tests/tasks/test_g1_antifall_export_contract.py`
- Modify: `tests/tasks/test_g1_antifall_curriculum_runner.py`

**Step 1: Write failing tests**
- Assert a new `deploy/robots/g1_antifall/config/config.yaml` exposes an `AntiFall` FSM mode and points at an anti-fall policy directory.
- Assert the anti-fall deploy contract builder returns flat deploy observations using velocity-command aliases, no `gait_phase`, and `history_length == 3` for all actor terms.
- Assert curriculum policy-copy logic also propagates `params/deploy.yaml`.

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/scripts/test_g1_antifall_deploy_config.py tests/tasks/test_g1_antifall_export_contract.py tests/tasks/test_g1_antifall_curriculum_runner.py`
Expected: FAIL because the new deploy target and deploy contract writer do not exist yet.

### Task 2: Add Python anti-fall deploy export support

**Files:**
- Create: `src/tasks/velocity/rl/antifall_deploy_contract.py`
- Create: `src/tasks/velocity/rl/antifall_runner.py`
- Modify: `src/tasks/velocity/config/g1_antifall/__init__.py`
- Modify: `src/tasks/velocity/rl/curriculum_runner.py`

**Step 1: Implement the anti-fall deploy contract builder**
- Reuse `get_base_metadata(env, run_name)`.
- Map training actor terms to deploy names via aliases: `command -> velocity_commands`, `joint_pos -> joint_pos_rel`, `joint_vel -> joint_vel_rel`, `actions -> last_action`.
- Emit a flat `observations:` block compatible with single-input `obs` ONNX models.
- Preserve anti-fall history length 3 and omit `gait_phase`.

**Step 2: Implement an anti-fall runner subclass**
- Extend `VelocityOnPolicyRunner.save()` behavior by writing `params/deploy.yaml` next to exported `policy.onnx`.

**Step 3: Update anti-fall task registration**
- Register anti-fall stages/benchmark with the new runner subclass.

**Step 4: Keep curriculum top-level exports deployable**
- Copy `stages/*/params/deploy.yaml` into the top-level curriculum run whenever the latest stage policy is copied.

### Task 3: Add the dedicated `deploy/robots/g1_antifall` target

**Files:**
- Create: `deploy/robots/g1_antifall/CMakeLists.txt`
- Create: `deploy/robots/g1_antifall/main.cpp`
- Create: `deploy/robots/g1_antifall/include/Types.h`
- Create: `deploy/robots/g1_antifall/src/State_RLBase.cpp`
- Create: `deploy/robots/g1_antifall/config/config.yaml`
- Create: `deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/params/deploy.yaml`

**Step 1: Reuse the existing G1 deploy runtime layout**
- Mirror the G1 deploy folder structure.
- Reuse shared deploy headers and the support-geometry helper source from `deploy/robots/g1` rather than inventing a new runtime.

**Step 2: Create a dedicated anti-fall FSM**
- Keep `Passive`, `FixStand`, and one `AntiFall` RLBase state.
- Point `AntiFall.policy_dir` to `config/policy/antifall/stage4b/v0`.
- Provide keyboard/gamepad hints that match the reduced state set.

**Step 3: Seed a deploy YAML template**
- Match the anti-fall export contract: flat terms, aliased deploy names, history length 3, no gait phase.

### Task 4: Verify build/test behavior

**Files:**
- Modify only if verification reveals breakage.

**Step 1: Run targeted tests**

Run: `pytest -q tests/scripts/test_g1_antifall_deploy_config.py tests/tasks/test_g1_antifall_export_contract.py tests/tasks/test_g1_antifall_curriculum_runner.py`
Expected: PASS.

**Step 2: Build the new deploy target**

Run: `cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build && cmake --build deploy/robots/g1_antifall/build -j4`
Expected: build completes and produces the anti-fall controller binary.

**Step 3: Smoke-check the CLI**

Run: `deploy/robots/g1_antifall/build/g1_antifall_ctrl --help`
Expected: help text prints without runtime errors.
