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
python tools/play_antifall.py \
  --task Unitree-G1-AntiFall-Stage4b \
  --run-dir logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/05_stage4b \
  --checkpoint model_*.pt
```

`play_antifall.py` uses the native MuJoCo viewer.  Dragging the robot body in the
viewer applies interactive perturbations and is the fastest way to inspect
recovery behavior.

## Sim2Real cautions

### Policy bundle

The deploy target reads:

```text
deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/
  exported/policy.onnx
  params/deploy.yaml
```

Copy both files from the same exported stage run, for example:

```bash
mkdir -p deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/exported
mkdir -p deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/params
cp logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/05_stage4b/policy.onnx \
  deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/exported/policy.onnx
cp logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/05_stage4b/params/deploy.yaml \
  deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/params/deploy.yaml
```

If you trained `Unitree-G1-AntiFall-Stage4b` directly, use that run directory
instead of the curriculum `stages/05_stage4b` directory.

### SDK and build

Prerequisites are the same as base G1 Velocity:

- Unitree SDK2 installed under `/opt/unitree_robotics`.
- `libyaml-cpp-dev`, `libboost-all-dev`, `libeigen3-dev`, `libspdlog-dev`,
  `libfmt-dev`, and `zlib1g-dev`.
- Vendored ONNX Runtime under `deploy/thirdparty/onnxruntime-linux-*-1.22.0/`.

Build simulator and controller:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4

cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build
cmake --build deploy/robots/g1_antifall/build -j4
```

### Loopback startup

Terminal 1:

```bash
./simulate/build/unitree_mujoco --network lo
```

Terminal 2:

```bash
./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo --keyboard
```

Keyboard operation:

- `f`: Passive → FixStand.
- `v`: FixStand → AntiFall.
- `w/s`: forward/backward command.
- `a/d`: strafe left/right.
- `q/e`: turn left/right.
- release movement keys: stop command.
- `p`: return to Passive.

Joystick operation:

- `L2 + Up`: Passive → FixStand.
- `R2 + A`: FixStand → AntiFall.
- `L2 + B`: return to Passive.

### Real robot startup

After Python play and loopback simulator recovery checks pass:

```bash
./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=<robot_nic> --keyboard
```

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
repeatable checks.  `tools/benchmark_antifall.py` remains the shell-facing CLI,
while `tools/antifall_harness.py` contains reusable helper logic.
