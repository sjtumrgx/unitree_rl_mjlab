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

1. Build simulator and controller.
2. Start with loopback networking (`--network lo`).
3. Keep stale DDS processes from previous runs out of the test by killing old
   simulator/controller processes before launch.
4. Start with low command speeds and conservative modes.
5. If behavior diverges from Python play, compare:
   - joint order
   - action order
   - default pose
   - startup blend
   - command frame
   - depth/camera validity

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
