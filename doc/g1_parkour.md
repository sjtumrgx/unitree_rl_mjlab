# Unitree G1 Parkour Deploy Notes

This document tracks the dedicated **G1 parkour deploy/runtime lane** added for the
InstinctLab depth-conditioned parkour model.

## Runtime ownership

The parkour deploy/runtime artifacts now live under:

- `deploy/robots/g1_parkour/`
- `deploy/robots/g1_parkour/config/policy/parkour/v0/`

The staged bundle contains:

- `exported/actor.onnx`
- `exported/0-depth_encoder.onnx`
- `params/deploy.yaml`
- `params/env.yaml`
- `params/agent.yaml`
- `parkour_artifacts.json`

## Deploy contract summary

- dedicated FSM mode: `Parkour`
- proprio history length: `8`
- depth stack: `8 x 18 x 32`
- raw simulator depth resolution: `64 x 36`
- crop region: `[18, 0, 16, 16]`
- source history length: `37`
- frame skip: `5`
- DDS pointcloud topic: `rt/parkour_depth/points`

The deploy runtime uses a two-stage ONNX flow:

1. `0-depth_encoder.onnx`
2. `actor.onnx`

## Build

Controller:

```bash
cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build
cmake --build deploy/robots/g1_parkour/build -j4
```

Simulator:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4
```

## Run

Start the parkour simulator:

```bash
simulate/build/unitree_mujoco_parkour --network=lo
```

Start the parkour controller in a second terminal:

```bash
deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo --keyboard
```

Keyboard transitions:

- `f` -> `FixStand`
- `k` -> `Parkour`
- `p` -> `Passive`
- `w/s/a/d/q/e` -> command velocity while in parkour mode

Gamepad transitions:

- `L2 + Up` -> `FixStand`
- `R2 + X` -> `Parkour`
- `L2 + B` -> `Passive`

## Simulator notes

`unitree_mujoco_parkour` now:

- loads `simulate/config_parkour.yaml`
- uses `scene_g1_parkour.xml`
- publishes organized PointCloud2 from the torso-mounted `parkour_depth_camera`
- shows an extra grayscale depth window for camera debugging

## Current limitation

The new lane is validated for the local MuJoCo + DDS simulator path.

The full real-robot depth integration path still needs a dedicated hardware bring-up pass
if this controller is moved from simulator DDS pointclouds to a live onboard depth source.
