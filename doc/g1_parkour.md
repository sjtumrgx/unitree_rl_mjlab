# G1 Parkour: Train → Play → Sim2Real Notes

G1 Parkour is an added depth-conditioned module for an exported InstinctLab-style
policy.  The current lane focuses on replay, terrain editing, deploy contract
validation, and Unitree C++/DDS simulation/runtime integration.

## Artifact contract

Default policy bundle:

```text
deploy/robots/g1_parkour/config/policy/parkour/v0/
```

Important files:

- `exported/actor.onnx`
- `exported/0-depth_encoder.onnx`
- `params/deploy.yaml`
- `parkour_artifacts.json`

The deploy YAML is the contract between Python and C++:

- training/deploy joint order
- action size and action scale
- depth camera name
- depth crop size and offset
- normalized depth range
- depth history shape

Do not update ONNX files without updating the matching deploy YAML and manifest.

## Play

Default visual replay:

```bash
python scripts/play_parkour.py
```

Contract-only check:

```bash
python scripts/play_parkour.py --check-contract --viewer none --no-depth-viewer
```

Short headless walk validation:

```bash
python scripts/play_parkour.py --validate-walk --viewer none --no-depth-viewer --max-steps 20
```

`terrain-route` mode reads `g1_parkour_route_waypoints` from
`src/tasks/velocity/config/g1_parkour/env_cfgs.py` and generates commands that
follow the terrain sequence.  If the terrain layout changes, update these
waypoints.

## Terrain editing

The editable terrain source of truth is:

```text
src/assets/robots/unitree_g1/xmls/scene_g1_parkour.xml
```

Open the browser editor:

```bash
python scripts/edit_parkour_scene.py --open-browser
```

The editor shows full dimensions.  MuJoCo XML stores box half-extents, so a full
`0.36 m` length is stored as `size="0.18 ..."`.  The Python MJLab debug spec reads
modules from the same XML, so saved XML changes affect both standalone simulator
scene loading and `scripts/play_parkour.py` after restart.

## Sim2Real / C++ runtime

Build:

```bash
cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build
cmake --build deploy/robots/g1_parkour/build -j4
```

Loopback simulator run:

```bash
./simulate/build/unitree_mujoco_parkour --network lo
./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo
```

Useful diagnostics:

- `G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5` on the controller isolates locomotion
  from live depth.
- `G1_PARKOUR_DEPTH_BRIDGE=0` on the simulator disables simulator-side live depth
  publishing.
- `--sim-autostart-parkour` starts directly in Parkour for automated simulator
  checks.
- `--no-sim-route-follow` disables route following for command ablations.

## Common failure modes

| Symptom | First checks |
| --- | --- |
| Walks on constant depth but fails terrain | Depth crop/range/history, camera pose, live-depth validity. |
| Python play works but C++ loopback fails | Joint order, action order, startup blend, default pose, PD bridge mode. |
| Robot drifts with zero command | Idle command and command-frame conventions. |
| Terrain edit looks right but route fails | Update `g1_parkour_route_waypoints`. |
| Viewer/depth window freezes | Hide depth debug window, use EGL/headless mode, or disable depth bridge. |
