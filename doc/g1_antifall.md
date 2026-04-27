# G1 AntiFall: Train → Play → Sim2Real Notes

AntiFall is an added G1 module on top of the Unitree RL MJLab baseline.  It aims
to make a proprioceptive G1 policy recover from pushes, kicks, and near-fall
states without changing the deploy-side observation/action contract.

## Train

Recommended entrypoint:

```bash
python scripts/train.py Unitree-G1-AntiFall-Curriculum \
  --env.scene.num-envs=4096 \
  --runner.max_iterations=5000
```

The curriculum exposes these stages for manual ablation:

| Stage | Meaning |
| --- | --- |
| `Stage0` | Standing/walking seed with no external disturbance. |
| `Stage1` | Flat push/kick recovery. |
| `Stage2` | Harder recovery and near-failure reset states. |
| `Stage3` | Walking-biased push/kick recovery. |
| `Stage4a` | Lateral/off-center disturbance recovery. |
| `Stage4b` | Hardest mixed standing/walking recovery. |

Use `Unitree-G1-AntiFall-Curriculum` for normal training.  Use the stage tasks
only when you intentionally want a single-stage run.

Stage checkpoints are stored under:

```text
logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/<stage>/model_*.pt
```

## Play

Replay a stage checkpoint, not the top-level exported ONNX:

```bash
python scripts/play_antifall.py \
  --task Unitree-G1-AntiFall-Stage4b \
  --run-dir logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/05_stage4b \
  --checkpoint model_*.pt
```

`play_antifall.py` uses the native MuJoCo viewer.  Dragging the robot body in the
viewer applies interactive perturbations and is the fastest way to inspect
recovery behavior.

## Sim2Real cautions

- Validate each stage in Python before exporting or deploying.
- Do not use AntiFall to hide incorrect action order, weak PD settings, or bad
  reset poses.  Fix contract errors first.
- Start hardware checks with low command speed and a safety operator ready to
  switch to Passive.
- If the policy recovers in Python but not in C++/DDS, compare default joint
  pose, action scale, command frame, and observation ordering before changing
  reward or curriculum logic.

## Benchmark helpers

`Unitree-G1-AntiFall-Benchmark` is a deterministic environment config for
repeatable checks.  `scripts/benchmark_antifall.py` remains the shell-facing CLI,
while `scripts/antifall_harness.py` contains reusable helper logic.
