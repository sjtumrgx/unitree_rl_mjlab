# G1 AntiFall-GetUp: Train → Play → Sim2Real Notes

AntiFall-GetUp is a combined G1 policy lane: it keeps the Stage4b AntiFall
walking/push-recovery prior and adds a GetUp-style fallen-start recovery branch.
It is a separate MJLab training/play task, but it does not currently have a
separate C++ controller target.

## Train

Train the two priors first:

```bash
python scripts/train.py Unitree-G1-AntiFall-Curriculum \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10000

python scripts/train_getup.py --terrain mixed -- \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001
```

Then train the fallen-start recovery warmup under the final AntiFall-GetUp tensor
contract:

```bash
python scripts/train.py Unitree-G1-AntiFall-GetUp-RecoveryWarmup \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001 \
  --agent.resume=True \
  --agent.actor-only-resume=True \
  --resume-checkpoint-path logs/rsl_rl/g1_getup/<getup_run>/model_*.pt \
  --agent.run-name recovery_warmup
```

Finally fuse the walking and recovery priors:

```bash
python scripts/train.py Unitree-G1-AntiFall-GetUp \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001 \
  --agent.resume=True \
  --agent.actor-only-resume=True \
  --resume-checkpoint-path logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/05_stage4b/model_*.pt \
  --recovery-resume-checkpoint-path logs/rsl_rl/g1_antifall_getup/<recovery_run>/model_*.pt
```

The final run writes under:

```text
logs/rsl_rl/g1_antifall_getup/<run>/
  model_*.pt
  policy.onnx
  params/deploy.yaml
```

## Play

Use the generic play entrypoint:

```bash
python scripts/play.py Unitree-G1-AntiFall-GetUp \
  --checkpoint_file logs/rsl_rl/g1_antifall_getup/<run>/model_*.pt \
  --num_envs 1 \
  --viewer native
```

In the native MuJoCo viewer, check three phases before considering C++ work:

- normal walking command tracking while upright;
- disturbance absorption without falling;
- fallen recovery and return to controllable locomotion.

## Sim2Real / current deploy gate

AntiFall-GetUp currently has no dedicated C++ binary such as
`g1_antifall_getup_ctrl`.  The only plausible deploy lane is to reuse the
29-DoF AntiFall controller:

```text
deploy/robots/g1_antifall/
```

However, the final AntiFall-GetUp exported `params/deploy.yaml` can include
additional observation terms such as `getup_progress`, `bfm_local_body_state`,
`height_scan`, and `recovery_phase`.  The current shared C++ observation registry
does not register those terms, so copying the final AntiFall-GetUp bundle into
`deploy/robots/g1_antifall/config/policy/antifall/stage4b/v0/` is not a
hardware-ready path.

Do not run AntiFall-GetUp on hardware until one of these is true:

- the final exported deploy YAML only uses C++-registered observations; or
- the missing C++ deploy observations/actions are implemented and a loopback
  startup proves the policy loads and runs.

## SDK and build once the deploy contract is supported

Prerequisites are the same as AntiFall:

- Unitree SDK2 installed under `/opt/unitree_robotics`;
- `libyaml-cpp-dev`, `libboost-all-dev`, `libeigen3-dev`, `libspdlog-dev`,
  `libfmt-dev`, and `zlib1g-dev`;
- vendored ONNX Runtime under `deploy/thirdparty/onnxruntime-linux-*-1.22.0/`.

Build simulator and the reusable AntiFall controller:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4

cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build
cmake --build deploy/robots/g1_antifall/build -j4
```

After the C++ deploy contract is supported, the loopback command shape is:

```bash
./simulate/build/unitree_mujoco --network lo
./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo --keyboard
```

Controls are inherited from AntiFall:

- `f`: Passive → FixStand.
- `v`: FixStand → RL control.
- `w/s/a/d/q/e`: velocity commands while upright.
- `p`: return to Passive.
- joystick `L2 + Up`: Passive → FixStand.
- joystick `R2 + A`: FixStand → RL control.
- joystick `L2 + B`: return to Passive.

Only after the loopback run loads the final AntiFall-GetUp deploy YAML, runs
without missing observation errors, and recovers from a forced-fall test should
the hardware command shape be considered:

```bash
./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=<robot_nic> --keyboard
```
