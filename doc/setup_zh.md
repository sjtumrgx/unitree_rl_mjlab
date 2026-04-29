# 环境配置

本仓库基于 Unitree RL MJLab，并额外增加 Parkour、AntiFall、GetUp 模块。建议 Train
→ Play → Sim2Real 全流程使用同一个 Python 环境，以保证 Python 回放和 C++ 部署读取
同一套 policy contract。

## 1. 主机要求

推荐配置：

- Ubuntu 22.04
- Python 3.11
- 大规模训练建议使用 NVIDIA GPU
- 使用 CUDA/MuJoCo 渲染时建议 NVIDIA driver 550+
- MuJoCo native viewer 和深度渲染需要可用 OpenGL/EGL 环境

无显示服务器可以通过 `--viewer none`、`--no-depth-viewer` 或 `MUJOCO_GL=egl` 运行
多数验证命令。

## 2. Python 环境

```bash
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab

git clone https://github.com/sjtumrgx/unitree_rl_mjlab.git
cd unitree_rl_mjlab
pip install -e .
```

核心 Python 依赖固定在 `setup.py`：

- `mjlab==1.2.0`
- `mujoco-warp==3.5.0`

如果使用 Viser 或 Matplotlib 等额外可视化窗口，请在同一环境中安装缺失的可选依赖。

## 3. C++/DDS 系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

这些依赖用于 Unitree MuJoCo simulator 和 C++ deploy controller。

## 4. 构建 simulator 和 deploy controller

Simulator：

```bash
cd simulate
cmake -B build -S .
cmake --build build -j4
cd ..
```

G1 Parkour controller：

```bash
cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build
cmake --build deploy/robots/g1_parkour/build -j4
```

其他 deploy controller 也按 `deploy/robots/<robot>/` 下的相同方式构建。

## 5. 显示与渲染注意事项

- MuJoCo native viewer 需要 `DISPLAY` 或 `WAYLAND_DISPLAY`。
- 无头服务器上，Python play 优先使用 `--viewer none --no-depth-viewer`。
- 离屏 MuJoCo 渲染可在导入 MuJoCo 前设置 `MUJOCO_GL=egl`。
- Parkour 深度诊断依赖躯干上的 `parkour_depth_camera`；如果深度窗口能打开但策略行为异常，优先检查 deploy YAML 的 crop/range/history，而不是直接调 controller gain。

## 6. 快速检查命令

```bash
# Python import / task registration 检查。
python scripts/list_envs.py | grep Unitree-G1

# Parkour contract 检查。
python scripts/play_parkour.py --check-contract --viewer none --no-depth-viewer

# Parkour 无头短回放验证。
python scripts/play_parkour.py --validate-walk --viewer none --no-depth-viewer --max-steps 20
```

建议在 C++/DDS 或真实机器人测试前先跑这些检查。
