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
python tools/play_parkour.py
```

Contract-only check:

```bash
python tools/play_parkour.py --check-contract --viewer none --no-depth-viewer
```

Short headless walk validation:

```bash
python tools/play_parkour.py --validate-walk --viewer none --no-depth-viewer --max-steps 20
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
python tools/edit_parkour_scene.py --open-browser
```

The editor shows full dimensions.  MuJoCo XML stores box half-extents, so a full
`0.36 m` length is stored as `size="0.18 ..."`.  The Python MJLab debug spec reads
modules from the same XML, so saved XML changes affect both standalone simulator
scene loading and `tools/play_parkour.py` after restart.

## Sim2Real / C++ runtime

### SDK and build

Prerequisites:

- Unitree SDK2 installed under `/opt/unitree_robotics`.
- System packages from setup docs.
- Vendored ONNX Runtime under `deploy/thirdparty/onnxruntime-linux-*-1.22.0/`.
- `simulate/config_parkour.yaml` points to
  `src/assets/robots/unitree_g1/xmls/scene_g1_parkour.xml`.

Build simulator and controller:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4

cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build
cmake --build deploy/robots/g1_parkour/build -j4
```

### Loopback startup

Use two terminals.

Terminal 1:

```bash
./simulate/build/unitree_mujoco_parkour --network lo
```

Terminal 2:

```bash
./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo
```

With `--network=lo`, the controller defaults to an interactive loopback mode:
it starts in Parkour idle-hold, enables live depth, and gates route following on
keyboard input.

Keyboard operation:

- hold `w` or `up`: follow the parkour terrain route at the configured cruise
  speed.
- release `w` / `up`: stop back to idle-hold.
- `+` / `=` and `-`: adjust held-walk speed.
- `a` / `left` / `q`: turn left.
- `d` / `right` / `e`: turn right.
- `c`: stop yaw turn.
- `s` / `down` / `x` / `space`: return to idle-hold command.
- `p`: return to Passive.

For automated loopback checks:

```bash
./deploy/robots/g1_parkour/build/g1_parkour_ctrl \
  --network=lo \
  --sim-autostart-parkour
```

Useful diagnostics:

- `G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5` on the controller isolates locomotion
  from live depth.
- `G1_PARKOUR_DEPTH_BRIDGE=0` on the simulator disables simulator-side live depth
  publishing.
- `--sim-autostart-parkour` starts directly in Parkour for automated simulator
  checks.
- `--no-sim-route-follow` disables route following for command ablations.
- `--constant-depth <value>` is the command-line equivalent of the constant-depth
  diagnostic.

### Real robot startup

Hardware command shape:

```bash
./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=<robot_nic> --keyboard
```

Joystick operation:

- `L2 + Up`: Passive → FixStand.
- `R2 + X`: FixStand → Parkour.
- `L2 + B`: return to Passive.

Do not use `--sim-autostart-parkour` on hardware; it is loopback-only and rejects
non-`lo` network interfaces.  Confirm live depth validity before moving from
constant-depth ablation to terrain traversal.

## Common failure modes

| Symptom | First checks |
| --- | --- |
| Walks on constant depth but fails terrain | Depth crop/range/history, camera pose, live-depth validity. |
| Python play works but C++ loopback fails | Joint order, action order, startup blend, default pose, PD bridge mode. |
| Robot drifts with zero command | Idle command and command-frame conventions. |
| Terrain edit looks right but route fails | Update `g1_parkour_route_waypoints`. |
| Viewer/depth window freezes | Hide depth debug window, use EGL/headless mode, or disable depth bridge. |
