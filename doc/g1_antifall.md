# Unitree G1 Anti-Fall Task Notes

This branch wires the anti-fall helper stack into the actual `g1_antifall` task configs while preserving the existing `Unitree-G1-Flat` baseline and the blind/proprioceptive actor contract.

## What is wired now

- **Actor contract** remains limited to proprioceptive terms:
  - `base_ang_vel`
  - `projected_gravity`
  - `command`
  - `joint_pos`
  - `joint_vel`
  - `actions`
- **Critic context** now includes disturbance-aware helper terms:
  - `disturbance_metadata`
  - `recovery_features`
- **Rewards** now include anti-fall shaping:
  - `upright_recoverability`
  - `recovery_quality`
  - `standing_stability`
  - `recovery_completion_bonus`
- **Metrics** now report disturbance/recovery signals:
  - `disturbance_window_active`
  - `disturbance_magnitude`
  - `controllable_locomotion`
  - `disturbance_count`
  - `recovery_success_count`
  - `recovery_latency`
- **Events**
  - push-based stages use `push_by_setting_velocity_with_history`
  - all anti-fall stages use `reset_root_state_mixed`
  - Stage 2+ can inject near-failure reset starts through the mixed-reset helper

## Stage semantics

| Stage | Surface | Disturbance / hazard |
| --- | --- | --- |
| Stage0 | Flat | No external disturbance |
| Stage1 | Flat | Mild tracked push recovery |
| Stage2 | Flat | Harder tracked pushes + occasional near-failure reset starts |
| Stage3 | Flat | Walking-biased push / kick recovery + occasional near-failure reset starts |
| Stage4a | Flat | Lateral / asymmetric push-kick recovery + occasional near-failure reset starts |
| Stage4b | Flat | Hardest mixed standing / walking push-kick recovery + occasional near-failure reset starts |
| Benchmark | Flat deterministic | Randomization disabled for reproducible evaluation |

## Automatic curriculum entrypoint

The repo now also exposes `Unitree-G1-AntiFall-Curriculum`, which keeps the
existing stage tasks intact but runs them through one top-level curriculum
process:

`Stage0 -> Stage1 -> Stage2 -> Stage3 -> Stage4a -> Stage4b`

Current default behavior:
- per-stage fallback budget: `10000` iterations
- Stage0 promotion gate: controllable locomotion threshold
- Stage1-Stage4b promotion gate: recovery-rate + recovery-latency threshold
- the late-stage curriculum remains a flat push-kick ladder, so stage promotion no longer depends on rough/slip/trip-specific hazard families
- every stage transition is written to `curriculum_manifest.json` in the run dir

Example:

```bash
python scripts/train.py Unitree-G1-AntiFall-Curriculum --env.scene.num-envs=4096
```

The original `Unitree-G1-AntiFall-Stage0` ... `Stage4b` tasks remain available
for manual training and debug workflows.

> Migration note: checkpoints produced before this push-kick semantic reset may still
> be loadable, but late-stage runs from that era encode the older rough/slip/trip
> semantics rather than the current mainline ladder.

## Benchmark CLI

Use the compatibility wrapper:

```bash
python scripts/benchmark_antifall.py scenarios Unitree-G1-AntiFall-Benchmark
python scripts/benchmark_antifall.py smoke-command Unitree-G1-AntiFall-Stage4b --seed 7
python scripts/benchmark_antifall.py training-health path/to/train.log
```

`scripts/antifall_harness.py` remains the library surface; `scripts/benchmark_antifall.py` is the stable entrypoint for shell workflows.

## Remaining gaps

- No long-horizon training run was completed in this worker lane; only config/import/CLI smoke verification was performed.
- Export/deploy validation is still limited to config/registry/command-contract checks, not a full policy export artifact.
