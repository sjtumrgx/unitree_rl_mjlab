<h1 align="center">Unitree RL MJLab：Train → Play → Sim2Real</h1>

<p align="center">
  <a href="README.md">English</a> · <a href="README_zh.md">中文</a>
</p>

本仓库基于 **Unitree RL MJLab / mjlab**，保留原有以 MuJoCo 为核心的训练、回放
和部署流程。在基础 Unitree locomotion 示例之上，本仓库额外增加了 **G1 Parkour**、
**G1 AntiFall** 和 **G1 AMP-Locomotion** 模块，并补充了面向 Unitree C++/DDS 控制链的仿真与
部署胶水代码。

推荐工作流是：

```text
1. Train      -> 在 MJLab/MuJoCo 中训练或适配策略
2. Play       -> 在 Python/MuJoCo 中回放、观察和排查策略
3. Sim2Real   -> 先经过 Unitree simulator/C++ controller 验证，再上真实机器人
```

## 仓库结构

```text
.
├── src/
│   ├── tasks/velocity/config/        # MJLab velocity 与 AntiFall 任务注册/环境配置。
│   ├── tasks/velocity/mdp/           # Velocity/AntiFall 奖励、重置、事件和观测辅助逻辑。
│   ├── tasks/amp_loco/               # AMP-Locomotion 配置、AMP runner、恢复重置逻辑和指标。
│   └── parkour/                      # Parkour ONNX/deploy contract、观测适配、深度工具和场景编辑核心。
├── scripts/
│   ├── train.py                      # 通用训练入口。
│   └── play.py                       # 通用 Python 策略回放 / 可视化入口。
├── tools/                            # 从 scripts/ 移出的辅助 CLI 和诊断工具。
├── data/
├── deploy/
│   └── robots/g1_parkour/            # C++/DDS G1 Parkour controller 与 policy runtime。
├── simulate/                         # Unitree MuJoCo simulator 集成与配置。
└── doc/                              # 面向用户的环境、模块和 sim2real 文档。
```

## 环境配置

推荐主机：

- Ubuntu 22.04
- 用于高吞吐训练的 NVIDIA GPU
- 使用 CUDA/MuJoCo 渲染时建议 NVIDIA Driver 550+
- Python 3.11

创建并激活环境：

```bash
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab
```

安装 C++ simulator/deploy 侧依赖：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

安装 Python 包：

```bash
git clone https://github.com/sjtumrgx/unitree_rl_mjlab.git
cd unitree_rl_mjlab
pip install -e .
```

`setup.py` 固定了核心 MJLab 依赖（`mjlab==1.2.0`, `mujoco-warp==3.5.0`）。
如果在无显示机器上无法打开 native viewer 或 depth window，请使用
`--viewer none --no-depth-viewer`，或者正确设置 `DISPLAY` / `WAYLAND_DISPLAY` /
`MUJOCO_GL`。

## 1. Train

下面所有训练示例都显式包含：

- `--gpu-ids "[0]"`：指定使用 0 号 GPU
- `--env.scene.num-envs=4096`：指定并行环境数量
- `--agent.max-iterations=...` 或 wrapper 的 `--max-iterations ...`：指定最大训练迭代数

### 1.1 基础速度跟踪策略

通用训练脚本的第一个参数是任务 id：

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001
```

常见 flat velocity 任务：

- `Unitree-Go2-Flat`
- `Unitree-G1-Flat`
- `Unitree-G1-23Dof-Flat`
- `Unitree-H1_2-Flat`
- `Unitree-A2-Flat`
- `Unitree-R1-Flat`

常用参数：

```bash
# 用 dotted config key 覆盖 GPU、环境数量或训练预算。
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids "[1]" \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=3000
```

训练日志保存在 `logs/rsl_rl/<experiment>/<run>/`。

### 1.2 G1 AntiFall

AntiFall 为 G1 增加分阶段扰动恢复课程。它保持盲态/本体感知 actor contract，
但逐步增加恢复难度：

| 任务 | 用途 |
| --- | --- |
| `Unitree-G1-AntiFall-Stage0` | 平地站立/行走种子任务。 |
| `Unitree-G1-AntiFall-Stage1` | 平地抗推/抗踢恢复。 |
| `Unitree-G1-AntiFall-Stage2` | 更难恢复任务，包含近失稳重置初态。 |
| `Unitree-G1-AntiFall-Stage3` | 偏行走中的抗推/抗踢恢复。 |
| `Unitree-G1-AntiFall-Stage4a` | 偏侧向/偏心扰动恢复。 |
| `Unitree-G1-AntiFall-Stage4b` | 最难的站立/行走混合扰动任务。 |
| `Unitree-G1-AntiFall-Curriculum` | 推荐的自动课程学习入口。 |
| `Unitree-G1-AntiFall-Benchmark` | 确定性评测配置。 |

推荐训练入口：

```bash
python scripts/train.py Unitree-G1-AntiFall-Curriculum \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10000
```

对 curriculum 而言，`agent.max-iterations` 是每个 stage 的预算。Stage checkpoint
保存在：

```text
logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/<stage>/model_*.pt
```

顶层导出的 ONNX/policy artifact 面向部署；Python 回放应使用具体 stage checkpoint。
路径根目录跟随 runner 配置：curriculum checkpoint 使用
`logs/rsl_rl/g1_antifall_curriculum`，直接训练的 AntiFall stage 使用
`logs/rsl_rl/g1_antifall`。


### 1.3 G1 AMP-Locomotion

AMP-Locomotion 现在是唯一的跌倒恢复 locomotion 训练链路。它用一个 policy
同时学习 walk/run locomotion 和跌倒恢复：一部分 delayed-termination 环境触发
终止后不会立刻 reset，而是在 recovery window 内继续尝试恢复，并优先从 recovery
motion clip 重置；AMP discriminator 约束行走/奔跑和恢复动作都贴近运动先验。

已注册任务：

| Task | 地形 | 用途 |
| --- | --- | --- |
| `Unitree-G1-AMP-Flat` | 平面 | 推荐的首个完整训练目标 |
| `Unitree-G1-AMP-Rough` | rough curriculum | 同一 AMP/recovery contract 的粗糙地形版本 |

已从 `~/AMP_mjlab` 复制的动作数据布局：

- 原始 CSV：`motion_data_csv/amp/`
- Walk/run NPZ：`src/assets/motions/g1/amp/WalkandRun/`
- Recovery NPZ：`src/assets/motions/g1/amp/Recovery/`

如果当前环境还不支持 `history_ordering=time`，训练前应用 mjlab observation
history patch：

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  $(python - <<'PY'
import mjlab.managers.observation_manager as om
print(om.__file__)
PY
)
```

查看迁移后的任务：

```bash
python tools/list_envs.py --keyword AMP
```

按要求启动 4 卡训练：

```bash
python scripts/train.py Unitree-G1-AMP-Flat \
  --gpu-ids "[0,1,2,3]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=20000
```

日志写入 `logs/rsl_rl/g1_amp_locomotion/<run>/`。约 2w（20k）iterations
附近 policy 经常会突然学会跌倒恢复；这时 AMP/recovery 指标出现阶跃式变化通常是
预期现象，不应直接判定为训练异常。

本次迁移在当前 checkout 上完成的验证训练：

- 命令：`python scripts/train.py Unitree-G1-AMP-Flat --gpu-ids "[0,1,2,3]" --env.scene.num-envs=4096 --agent.max-iterations=20000 --agent.save-interval=100 --agent.logger=tensorboard`
- Run 目录：`logs/rsl_rl/g1_amp_locomotion/2026-06-02_14-47-05/`
- 最终 checkpoint：`model_19999.pt`（0-based 日志共 20,000 次迭代：`0..19999`）
- 导出策略：`policy.onnx`
- 最终指标快照：mean reward `41.03`，mean episode length `1000.0`，`bad_orientation=0.0`，`bad_base_height=0.0`，`skipped_non_finite_batches=0.0`
- Launcher 结果：torchrunx 报告 `Workers exited without errors`。

`logs/` 下的训练产物是本机输出，按仓库规则不纳入版本控制；在其他机器上需要重新运行上面的命令生成 checkpoint。

### 1.4 G1 Parkour artifact

当前 Parkour 主要是一个 **play/deploy lane**，用于已导出的 InstinctLab 风格深度
策略。默认 policy bundle 位于：

```text
deploy/robots/g1_parkour/config/policy/parkour/v0/
```

如果要提升新的训练 artifact bundle，请保持 `actor.onnx`、`0-depth_encoder.onnx`、
`params/deploy.yaml` 和 `parkour_artifacts.json` 同步。`deploy.yaml` 定义了 Python
与 C++ runtime 共用的关节顺序、动作尺度、深度裁剪、深度范围和相机 contract。

## 2. Play

Play 阶段应该在进入 C++/DDS 或真实机器人之前尽量发现 transfer 问题。优先检查
关节顺序、动作顺序、命令坐标系、reset pose、viewer 与 depth 诊断。

### 2.1 通用 play

```bash
python scripts/play.py Unitree-G1-Flat \
  --checkpoint-file logs/rsl_rl/g1_velocity/<run>/model_*.pt
```

需要 MuJoCo 鼠标/键盘交互时使用 `--viewer native`；无头验证用 `--viewer none`。

### 2.2 AntiFall play

用 native MuJoCo viewer 回放具体 stage checkpoint：

```bash
python tools/play_antifall.py \
  --task Unitree-G1-AntiFall-Stage4b \
  --checkpoint-file logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/stages/05_stage4b/model_*.pt \
  --num-envs 1 \
  --device cuda:0
```

如果是直接训练 `Unitree-G1-AntiFall-Stage4b`，则改用
`logs/rsl_rl/g1_antifall/<stage4b_run>/model_*.pt`。


### 2.3 AMP-Locomotion play

使用通用 play 入口回放训练好的 AMP-Locomotion checkpoint：

```bash
python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run>/model_*.pt
```

粗糙地形版本使用 `Unitree-G1-AMP-Rough`。

### 2.4 Parkour play 与地形编辑
### 2.5 Parkour play 与地形编辑

默认 Parkour 回放：

```bash
python tools/play_parkour.py
```

无头验证：

```bash
python tools/play_parkour.py --validate-walk \
  --viewer none \
  --no-depth-viewer \
  --max-steps 20
```

默认命令模式是 `terrain-route`：脚本会从任务配置读取
`g1_parkour_route_waypoints`，生成沿确定性地形序列前进的速度命令。只有做固定命令
ablation 时才使用 `--command-mode fixed`。

Parkour 地形唯一来源现在是：

```text
src/assets/robots/unitree_g1/xmls/scene_g1_parkour.xml
```

打开可视化地形编辑器：

```bash
python tools/edit_parkour_scene.py --open-browser
```

编辑器显示的是直观的**完整**长/宽/高。MuJoCo XML 的 box `size` 是半尺寸，所以
界面中的 `0.36 x 1.44 x 0.04 m` 会写成 `size="0.18 0.72 0.02"`。如果修改障碍顺序
或总路线长度，需要同时更新 `src/tasks/velocity/config/g1_parkour/env_cfgs.py` 中的
route waypoints。

## 3. Sim2Real

Sim2Real 指从 Python/MJLab 回放进入 Unitree C++/DDS 控制链；先跑 simulator，再上
真实机器人。不同任务要分开看：simulator 是共用的，但 controller 二进制、policy
目录、键盘切换方式和安全门槛都不同。

### 3.1 通用 simulator 构建

先构建 Unitree MuJoCo simulator：

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j4
```

普通 simulator 二进制读取 `simulate/config.yaml`；Parkour 二进制读取
`simulate/config_parkour.yaml`，并默认启用 depth bridge。

### 3.2 各任务 loopback 命令

本机 loopback 使用两个终端：先启动 simulator，再用 `--network=lo` 启动对应
controller。真实机器人时，必须先通过 simulator 验证，再把同一个 controller 的
`lo` 替换成机器人网卡名。

| 任务 | 进入 C++ 前的 Python gate | Controller 构建 | Loopback simulator 终端 | Loopback controller 终端 | 进入/控制方式 | 主要注意事项 |
| --- | --- | --- | --- | --- | --- | --- |
| Velocity / 基础 G1 | `python scripts/play.py Unitree-G1-Flat --checkpoint_file <model.pt>` | `cmake -S deploy/robots/g1 -B deploy/robots/g1/build`<br>`cmake --build deploy/robots/g1/build -j4` | `./simulate/build/unitree_mujoco` | `./deploy/robots/g1/build/g1_ctrl --network=lo --keyboard` | 键盘：`f` → `v`；遥控器：`L2+Up` → `R2+A` | 部署 artifact 来自 `deploy/robots/g1/config/policy/velocity/v0`；重点核对 `params/deploy.yaml` 里的 joint order/action scale。 |
| AntiFall | `python tools/play_antifall.py --task Unitree-G1-AntiFall-Stage4b --run-dir <stage_dir> --checkpoint <model.pt>` | `cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build`<br>`cmake --build deploy/robots/g1_antifall/build -j4` | `./simulate/build/unitree_mujoco` | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo --keyboard` | 键盘：`f` → `v`；遥控器：`L2+Up` → `R2+A` | 先在 simulator 中验证恢复能力；不要用 AntiFall 掩盖 action order、PD gain 或 reset pose 错误。 |
| Parkour | `python tools/play_parkour.py --check-contract --viewer none --no-depth-viewer`<br>`python tools/play_parkour.py --validate-walk --viewer none --no-depth-viewer --max-steps 20` | `cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build`<br>`cmake --build deploy/robots/g1_parkour/build -j4` | `./simulate/build/unitree_mujoco_parkour` | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo`<br>自动进入：`./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo --sim-autostart-parkour` | loopback 键盘路线：按住 `w` / `up`；`p` 回 Passive | Live depth 是 policy contract 的一部分。常量深度只适合 ablation，不代表地形通过能力。 |

无头 simulator 诊断时，给 simulator 命令加 `--headless --headless-seconds <N>`。
Parkour 深度 ablation 可在 controller 侧设置 `G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5`；
如果要关闭 simulator 侧 live depth 发布，用 `G1_PARKOUR_DEPTH_BRIDGE=0` 启动 simulator。

### 3.3 真实机器人检查清单

上硬件前：

1. 用同一 checkpoint/artifact 通过 Python `play`。
2. 用上表中对应任务的保守命令通过 simulator loopback。
3. 确认 joint name、action scale、默认关节角和 policy observation order 与 deploy YAML 一致。
4. 确认机器人网卡和 DDS domain 不会连到旧 simulator/controller 进程。
5. 只有 simulator 验证通过后，才把 `--network=lo` 换成真实机器人网卡。
6. 低速开始，并安排安全员随时切 Passive。
7. 修改 action order、PD gain 或默认姿态后，首次电机使能应悬空或有支撑。
8. 不要通过盲目增大刚度来掩盖不稳定；优先检查 command frame、action order、torque limit、depth validity 和 reset pose。

Parkour 特别注意：

- Depth 是策略 contract 的一部分。常量深度只适合 ablation，不代表地形通过能力。
- 躯干上的 `parkour_depth_camera` 的 crop/range/history 必须和 `deploy.yaml` 一致。
- 修改 route waypoints 时要检查 Python route follower 和 C++ loopback 命令是否仍对齐。
- `scene_g1_parkour.xml` 的改动会影响 Python play，因为 MJLab debug spec 会从该 XML 读取地形模块。

## 更多文档

- `doc/setup_en.md` / `doc/setup_zh.md` — 环境配置。
- `doc/train_play_sim2real_en.md` / `doc/train_play_sim2real_zh.md` — 端到端流程检查清单。
- `doc/g1_velocity.md` — 基础 G1 velocity train/play/sim2real 说明。
- `doc/g1_antifall.md` — AntiFall train/play/sim2real 说明。
- `doc/g1_parkour.md` — Parkour artifact、深度、地形和部署说明。

## 致谢

本项目基于 Unitree RL MJLab、MJLab、MuJoCo、MuJoCo Warp、RSL-RL、Unitree SDK2
和 Unitree MuJoCo。本 fork 增加了 G1 Parkour、AntiFall、AMP-Locomotion 工作流及相关部署工具。
