# Environment Setup

This repository is a Unitree RL MJLab-based project with additional Parkour,
AntiFall, and GetUp modules.  Use the same environment for the Train → Play →
Sim2Real workflow so Python replay and C++ deployment consume the same policy
contracts.

## 1. Host requirements

Recommended:

- Ubuntu 22.04
- Python 3.11
- NVIDIA GPU for large training runs
- NVIDIA driver 550+ when using CUDA/MuJoCo rendering
- A working OpenGL/EGL stack for MuJoCo native viewer and depth rendering

Headless machines can still run most validation commands with `--viewer none`,
`--no-depth-viewer`, or `MUJOCO_GL=egl`.

## 2. Python environment

```bash
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab

git clone https://github.com/sjtumrgx/unitree_rl_mjlab.git
cd unitree_rl_mjlab
pip install -e .
```

Core Python dependencies are pinned in `setup.py`:

- `mjlab==1.2.0`
- `mujoco-warp==3.5.0`

If you use extra visualization tools such as Viser or Matplotlib windows, install
missing optional packages in the same environment.

## 3. System packages for C++/DDS

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev zlib1g-dev
```

These are needed by the Unitree MuJoCo simulator and C++ deploy controllers.

Install Unitree SDK2 into `/opt/unitree_robotics` before building the simulator
or controllers.  This repository expects the SDK headers and DDS libraries here:

```text
/opt/unitree_robotics/include/unitree/...
/opt/unitree_robotics/include/dds...
/opt/unitree_robotics/lib/libunitree_sdk2.a
/opt/unitree_robotics/lib/libddsc.so
/opt/unitree_robotics/lib/libddscxx.so
/opt/unitree_robotics/lib/cmake/unitree_sdk2/unitree_sdk2Config.cmake
```

If those files are missing, install or build Unitree SDK2 with the official
Unitree procedure first.  ONNX Runtime does not need a separate system install:
the x64/aarch64 runtimes are vendored under
`deploy/thirdparty/onnxruntime-linux-*-1.22.0/`, and each controller CMake file
selects the matching architecture.

## 4. Build simulator and deploy controllers

Simulator:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4
```

Build the controller for the method you are validating:

```bash
cmake -S deploy/robots/g1 -B deploy/robots/g1/build
cmake --build deploy/robots/g1/build -j4

cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build
cmake --build deploy/robots/g1_antifall/build -j4


cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build
cmake --build deploy/robots/g1_parkour/build -j4
```

If CMake cannot find `unitree_sdk2Config.cmake`, confirm that
`/opt/unitree_robotics/lib/cmake` exists.  If needed, pass it explicitly:

```bash
cmake -S simulate -B simulate/build \
  -DCMAKE_PREFIX_PATH=/opt/unitree_robotics/lib/cmake
```

After building, check that no runtime library is missing:

```bash
ldd simulate/build/unitree_mujoco | grep "not found" || true
ldd deploy/robots/g1/build/g1_ctrl | grep "not found" || true
```

## 5. Display and rendering notes

- Native MuJoCo viewers require `DISPLAY` or `WAYLAND_DISPLAY`.
- On headless servers, prefer `--viewer none --no-depth-viewer` for Python play.
- For offscreen MuJoCo rendering, set `MUJOCO_GL=egl` before importing MuJoCo.
- Parkour depth diagnostics use the torso-mounted `parkour_depth_camera`; if the
  depth window opens but policy behavior is wrong, verify the deploy YAML crop,
  range, and history settings before tuning controller gains.

## 6. Quick smoke commands

```bash
# Python import / task registration sanity.
python tools/list_envs.py | grep Unitree-G1

# Parkour contract check.
python tools/play_parkour.py --check-contract --viewer none --no-depth-viewer

# Headless short Parkour play validation.
python tools/play_parkour.py --validate-walk --viewer none --no-depth-viewer --max-steps 20
```

Use these before C++/DDS or real robot testing.
