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
| Stage3 | Rough terrain | Rough terrain + occasional near-failure reset starts |
| Stage4a | Flat low-friction | Slip-focused low-friction feet + occasional near-failure reset starts |
| Stage4b | Flat | Forward-biased tracked trip-like pushes + occasional near-failure reset starts |
| Benchmark | Flat deterministic | Randomization disabled for reproducible evaluation |

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
