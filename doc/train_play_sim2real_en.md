# Train → Play → Sim2Real Checklist

Use this checklist for every policy lane in this repository.  The goal is to
catch contract and runtime mismatches before they reach a real robot.

## 1. Train

1. Pick the task id and confirm the module maturity:
   - Base velocity and GetUp are trainable MJLab tasks.
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
   - GetUp: `scripts/play_getup.py`
   - Parkour: `scripts/play_parkour.py`
3. Check reset pose, command direction, action scale, joint order, and viewer
   diagnostics.
4. For Parkour, validate depth contract with `--check-contract` and use
   `--validate-walk --viewer none --no-depth-viewer` for headless smoke checks.
5. Only move forward when Python play matches the intended behavior.

## 3. Simulator / C++ loopback

1. Build the shared simulator once:

   ```bash
   cmake -S simulate -B simulate/build
   cmake --build simulate/build -j4
   ```

2. Build the controller for the exact task you are validating.
3. Start with loopback networking (`--network=lo`).
4. Keep stale DDS processes from previous runs out of the test by killing old
   simulator/controller processes before launch.
5. Start with low command speeds and conservative modes.
6. If behavior diverges from Python play, compare:
   - joint order
   - action order
   - default pose
   - startup blend
   - command frame
   - depth/camera validity

### Task-specific command matrix

| Task | Controller build | Simulator terminal | Controller terminal | Control transition | Real-robot command shape |
| --- | --- | --- | --- | --- | --- |
| Velocity / base G1 | `cmake -S deploy/robots/g1 -B deploy/robots/g1/build`<br>`cmake --build deploy/robots/g1/build -j4` | `./simulate/build/unitree_mujoco` | `./deploy/robots/g1/build/g1_ctrl --network=lo --keyboard` | keyboard `f` → `v`; joystick `L2+Up` → `R2+A` | `./deploy/robots/g1/build/g1_ctrl --network=<robot_nic> --keyboard` |
| AntiFall | `cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build`<br>`cmake --build deploy/robots/g1_antifall/build -j4` | `./simulate/build/unitree_mujoco` | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo --keyboard` | keyboard `f` → `v`; joystick `L2+Up` → `R2+A` | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=<robot_nic> --keyboard` |
| GetUp | `cmake -S deploy/robots/g1_getup -B deploy/robots/g1_getup/build`<br>`cmake --build deploy/robots/g1_getup/build -j4` | `./simulate/build/unitree_mujoco` | `./deploy/robots/g1_getup/build/g1_getup_ctrl --network=lo --keyboard` | keyboard `f` → `g`; joystick `L2+Up` → `R2+Y` | `./deploy/robots/g1_getup/build/g1_getup_ctrl --network=<robot_nic> --keyboard` |
| Parkour | `cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build`<br>`cmake --build deploy/robots/g1_parkour/build -j4` | `./simulate/build/unitree_mujoco_parkour` | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo`<br>auto-start: `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo --sim-autostart-parkour` | loopback route: hold `w` / `up`; `p` returns Passive | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=<robot_nic> --keyboard` |

Use `--headless --headless-seconds <N>` on the simulator for no-GUI loopback
runs.  Parkour also supports `G1_PARKOUR_DEPTH_BRIDGE=0` on the simulator and
`G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5` or `--constant-depth <value>` on the
controller for depth ablations.

## 4. Real robot gate

Do not run hardware until the simulator path is stable.  Before enabling motors:

- Confirm the correct network interface and DDS domain.
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
