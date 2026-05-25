# G1 Velocity: Train → Play → Sim2Real Notes

G1 Velocity is the base proprioceptive locomotion lane for the 29-DoF G1
controller.  Its deploy bundle lives under `deploy/robots/g1/` and is also the
reference wiring used by AntiFall.

## Train

Default flat velocity training:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001
```

Rough-terrain training uses the same entrypoint with `Unitree-G1-Rough`.
Training logs are written under `logs/rsl_rl/<experiment>/<run>/`, and the
deployable ONNX is exported as `policy.onnx`.

## Play

Replay the checkpoint before touching C++/DDS:

```bash
python scripts/play.py Unitree-G1-Flat \
  --checkpoint_file logs/rsl_rl/<experiment>/<run>/model_*.pt
```

Check that forward/backward/lateral/yaw command directions match expectations
and that the policy stands without drift at zero command.

## Sim2Real / Unitree simulator

### Policy bundle

The controller reads this bundle:

```text
deploy/robots/g1/config/policy/velocity/v0/
  exported/policy.onnx
  params/deploy.yaml
```

When replacing a trained policy, copy the exported ONNX into the bundle:

```bash
mkdir -p deploy/robots/g1/config/policy/velocity/v0/exported
cp logs/rsl_rl/g1_velocity/<run>/policy.onnx \
  deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx
```

The base Velocity runner exports `policy.onnx`; the controller-side
`params/deploy.yaml` is the deploy contract for joint order, action scale,
default pose, PD gains, commands, and observations.  Only replace `deploy.yaml`
when you have generated and verified a matching deployment contract for the same
policy.

### SDK and build

Prerequisites:

- Unitree SDK2 installed under `/opt/unitree_robotics`.
- System packages from `doc/setup_en.md` or `doc/setup_zh.md`.
- Vendored ONNX Runtime under `deploy/thirdparty/onnxruntime-linux-*-1.22.0/`.

Build the shared simulator and G1 controller:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4

cmake -S deploy/robots/g1 -B deploy/robots/g1/build
cmake --build deploy/robots/g1/build -j4
```

### Loopback startup

Use two terminals.

Terminal 1:

```bash
./simulate/build/unitree_mujoco --network lo
```

Terminal 2:

```bash
./deploy/robots/g1/build/g1_ctrl --network=lo --keyboard
```

Keyboard operation:

- `f`: Passive → FixStand.
- `v`: FixStand → Velocity.
- `w/s`: forward/backward command.
- `a/d`: strafe left/right.
- `q/e`: turn left/right.
- release movement keys: stop command.
- `m`: play the configured dance mimic from Velocity.
- `p`: return to Passive.

Joystick operation:

- `L2 + Up`: Passive → FixStand.
- `R2 + A`: FixStand → Velocity.
- `R1 + A/B/Y/X`: dance transitions when configured.
- `L2 + B`: return to Passive.

### Real robot startup

After loopback is stable, replace `lo` with the robot network interface:

```bash
./deploy/robots/g1/build/g1_ctrl --network=<robot_nic> --keyboard
```

Start from low-speed commands.  Keep a safety operator ready to return to
Passive.  If behavior differs from Python play, compare `params/deploy.yaml`,
joint order, action scale, default pose, command frame, and PD settings before
changing the trained policy.
