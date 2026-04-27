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
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

These are needed by the Unitree MuJoCo simulator and C++ deploy controllers.

## 4. Build simulator and deploy controllers

Simulator:

```bash
cd simulate
cmake -B build -S .
cmake --build build -j4
cd ..
```

G1 Parkour controller:

```bash
cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build
cmake --build deploy/robots/g1_parkour/build -j4
```

Other deploy controllers follow the same pattern under `deploy/robots/<robot>/`.

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
python scripts/list_envs.py | grep Unitree-G1

# Parkour contract check.
python scripts/play_parkour.py --check-contract --viewer none --no-depth-viewer

# Headless short Parkour play validation.
python scripts/play_parkour.py --validate-walk --viewer none --no-depth-viewer --max-steps 20
```

Use these before C++/DDS or real robot testing.
