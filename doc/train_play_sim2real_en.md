# Train → Play → Sim2Real Checklist

Use this checklist for every policy lane in this repository.  The goal is to
catch contract and runtime mismatches before they reach a real robot.

## 1. Train

1. Pick the task id and confirm the module maturity:
   - Base velocity and AMP-Locomotion are trainable MJLab tasks.
   - AntiFall is trainable through staged/curriculum tasks.
   - Parkour currently uses a play/deploy-first exported depth policy lane.
2. Start from the generic training entrypoint unless a module wrapper exists:

   ```bash
   python scripts/train.py <TaskId> --env.scene.num-envs=4096
   ```

3. Keep command-line overrides in the experiment notes: device, terrain,
   number of envs, max iterations, and checkpoint selection criteria.
4. Distinguish artifacts:
   - `model_*.pt`: replay/resume checkpoints for Python play.
   - `policy.onnx` or `actor.onnx`: deployment artifact.
   - `deploy.yaml`: deploy-side contract for joint order, action scale, and
     sensor/depth settings.

## 2. Play

Before C++/DDS:

1. Replay the exact checkpoint/artifact in Python.
2. Use the task-specific play script when available:
   - AntiFall: `scripts/play_antifall.py`
   - AMP-Locomotion: use generic `scripts/play.py Unitree-G1-AMP-Flat` or `Unitree-G1-AMP-Rough`
   - Parkour: `scripts/play_parkour.py`
3. Check reset pose, command direction, action scale, joint order, and viewer
   diagnostics.
4. For Parkour, validate depth contract with `--check-contract` and use
   `--validate-walk --viewer none --no-depth-viewer` for headless smoke checks.
5. Only move forward when Python play matches the intended behavior.

## 3. Simulator / C++ loopback

1. Confirm the C++/DDS prerequisites:
   - Unitree SDK2 is installed under `/opt/unitree_robotics` with `unitree_sdk2`,
     `ddsc`, and `ddscxx` headers/libraries.
   - System packages include `cmake`, `libyaml-cpp-dev`, `libboost-all-dev`,
     `libeigen3-dev`, `libspdlog-dev`, `libfmt-dev`, and `zlib1g-dev`.
   - ONNX Runtime is vendored under `deploy/thirdparty/onnxruntime-linux-*-1.22.0/`;
     no system ONNX Runtime install is required.
2. Build the shared simulator once:

   ```bash
   cmake -S simulate -B simulate/build
   cmake --build simulate/build -j4
   ```

3. Build the controller for the exact task you are validating.
4. Use loopback networking (`--network=lo`) for local simulation.  On hardware,
   replace `lo` with the robot NIC name from `ip addr`.
5. Keep stale DDS processes from previous runs out of the test:

   ```bash
   pkill -f unitree_mujoco || true
   pkill -f g1_ctrl || true
   pkill -f g1_antifall_ctrl || true
   pkill -f g1_parkour_ctrl || true
   ```

6. Use two terminals: start the simulator first, then start the matching
   controller.
7. Start with low command speeds and conservative modes.
8. If behavior diverges from Python play, compare:
   - joint order
   - action order
   - default pose
   - startup blend
   - command frame
   - depth/camera validity

### Task-specific command matrix

| Task | Controller build | Simulator terminal | Controller terminal | Control transition | Real-robot command shape |
| --- | --- | --- | --- | --- | --- |
| Velocity / base G1 | `cmake -S deploy/robots/g1 -B deploy/robots/g1/build`<br>`cmake --build deploy/robots/g1/build -j4` | `./simulate/build/unitree_mujoco --network lo` | `./deploy/robots/g1/build/g1_ctrl --network=lo --keyboard` | keyboard `f` enters FixStand, `v` enters Velocity, `w/s/a/d/q/e` command motion, release to stop, `p` returns Passive; joystick `L2+Up` → `R2+A`, `L2+B` returns Passive | `./deploy/robots/g1/build/g1_ctrl --network=<robot_nic> --keyboard` |
| AntiFall | `cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build`<br>`cmake --build deploy/robots/g1_antifall/build -j4` | `./simulate/build/unitree_mujoco --network lo` | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo --keyboard` | keyboard `f` enters FixStand, `v` enters AntiFall, `w/s/a/d/q/e` command motion, `p` returns Passive; joystick `L2+Up` → `R2+A`, `L2+B` returns Passive | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=<robot_nic> --keyboard` |
| AMP-Locomotion | Python training/play lane; deploy controller is not introduced in this migration | `./simulate/build/unitree_mujoco --network lo` | Python play: `python scripts/play.py Unitree-G1-AMP-Flat --checkpoint-file <model.pt>` | train/play first; export `policy.onnx` from the AMP runner | Validate in Python before adding a C++ controller contract. |
| Parkour | `cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build`<br>`cmake --build deploy/robots/g1_parkour/build -j4` | `./simulate/build/unitree_mujoco_parkour --network lo` | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo`<br>auto-start: `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo --sim-autostart-parkour` | loopback defaults to Parkour idle-hold; hold `w` / `up` for route following, release to stop, `+/-` changes speed, `a/d/q/e` turns, `s/down/x/space` idles, `p` returns Passive; hardware joystick `L2+Up` → `R2+X` | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=<robot_nic> --keyboard` |

Use `--headless --headless-seconds <N>` on the simulator for no-GUI loopback
runs.  Parkour also supports `G1_PARKOUR_DEPTH_BRIDGE=0` on the simulator and
`G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5` or `--constant-depth <value>` on the
controller for depth ablations.

### Common post-launch sequence

1. Wait for the simulator MuJoCo window or headless logs before pressing
   controller keys.
2. The controller should print `Waiting for connection to robot...` and then
   `Connected to robot.`; if it reports that the lowcmd channel is already in
   use, stop the stale controller first.
3. Keyboard mode requires the controller terminal to keep focus; `--keyboard`
   exits in non-interactive stdin.
4. Enter FixStand before entering the RL state so Passive blends into the default
   standing pose.
5. On any abnormal jitter, pose mismatch, bad depth, or contact issue, press `p`
   or joystick `L2+B` to return to Passive before stopping the controller.

## 4. Real robot gate

Do not run hardware until the simulator path is stable.  Before enabling motors:

- Confirm the correct network interface and DDS domain; do not use
  `--network=lo` on the real robot.
- Confirm the controller is built from the same source revision as the policy
  contract.
- Use low speed and keep a safety operator ready for Passive.
- Support or suspend the robot for first tests after changing action order,
  default pose, or PD gains.
- Treat constant-depth or headless diagnostics as ablations, not proof of terrain
  traversal readiness.

## Done criteria by module

| Module | Train done | Play done | Sim2Real gate |
| --- | --- | --- | --- |
| Base velocity | Stable checkpoint and expected tracking metrics. | Python play walks with correct command direction. | C++/DDS simulator stable at low speed. |
| AntiFall | Curriculum/stage checkpoint recovers in training eval. | Native viewer drag perturbations recover. | Conservative sim/hardware disturbance tests only. |
| GetUp | Terrain-specific checkpoint converges for selected terrain. | Same terrain play gets up repeatedly. | Start from ground terrain before platform/wall/slope hardware tests. |
| Parkour | Exported artifact contract is complete. | Depth contract and route-following replay pass. | Simulator live-depth route before any real-depth hardware trial. |
