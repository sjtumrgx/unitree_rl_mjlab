# Unitree RL Mjlab


## ✳️ 概述

Unitree RL Mjlab 是一个基于 [mjlab](https://github.com/mujocolab/mjlab.git) 构建的强化学习项目，
使用 MuJoCo 作为物理仿真后端，当前支持 Unitree Go2, A2, As2, G1, R1, H1_2 和 H2 机器人。

Mjlab 结合了 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的成熟高层 API 与 
[MuJoCo](https://github.com/google-deepmind/mujoco_warp) 的高精度物理引擎，
为强化学习机器人研究与 Sim-to-Real（仿真到实机） 部署提供了一个轻量化、模块化的框架。

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 安装配置

安装和配置步骤请参考 [setup.md](doc/setup_zh.md)


## 🔁 流程概览

使用强化学习实现机器人运动控制的基本流程如下：

`训练` → `仿真验证` → `仿真到实机`

- **训练**: 在 MuJoCo 模拟环境中让机器人与环境交互，并通过奖励函数最大化学习策略。
- **仿真验证**: 加载训练好的策略进行回放，验证策略行为是否符合预期。
- **仿真到实机**: 将策略部署到物理机器人上，实现真实环境中的运动控制。


## 🛠️ 使用指南

### 1. 速度跟踪训练

运行以下命令进行速度跟踪训练：

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

多 GPU 训练：使用 --gpu-ids 扩展到多块 GPU：

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids "[0,1]" \
  --env.scene.num-envs=4096
```

- 第一个参数(如 Mjlab-Velocity-Flat-Unitree-G1)为必选参数，确定要启用的训练环境。可选：
  - Unitree-Go2-Flat
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> 更多有关详细说明，请参阅 mjlab 文档
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 1.1 G1 抗摔任务训练

当前仓库已经加入分阶段的 **Unitree G1 Anti-Fall** 任务族。该任务保持部署侧
actor 为纯本体感觉观测，同时在 critic 侧加入扰动/恢复上下文、恢复奖励和
benchmark 工具。

可用任务：

- `Unitree-G1-AntiFall-Stage0` — 平地站立/行走种子任务（无外部扰动）
- `Unitree-G1-AntiFall-Stage1` — 平地抗推 / 抗踢恢复
- `Unitree-G1-AntiFall-Stage2` — 更强的平地恢复 + 近失稳重置
- `Unitree-G1-AntiFall-Stage3` — 以行走中抗推 / 抗踢为主的平地恢复
- `Unitree-G1-AntiFall-Stage4a` — 偏侧向 / 偏心脚踢样式的恢复
- `Unitree-G1-AntiFall-Stage4b` — 最难的站立 / 行走混合抗推抗踢恢复
- `Unitree-G1-AntiFall-Benchmark` — 确定性 benchmark 配置
- `Unitree-G1-AntiFall-Curriculum` — 单入口自动课程学习任务（保留上述 stage 任务用于手工调试 / 对比实验）

推荐课程顺序：

`Stage0 → Stage1 → Stage2 → Stage3 → Stage4a → Stage4b`

正式训练命令（推荐统一入口）：

```bash
python scripts/train.py Unitree-G1-AntiFall-Curriculum \
  --gpu-ids "[0]" \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10000 \
  --agent.save-interval=100
```

参数说明：

- 第一个位置参数：训练任务 ID。
  - 推荐使用 `Unitree-G1-AntiFall-Curriculum` 执行自动课程学习。
  - 只有在你明确要手工跑某一阶段时，才改成 `Unitree-G1-AntiFall-Stage0` ~ `Unitree-G1-AntiFall-Stage4b`。
- `--gpu-ids`：训练使用的 GPU 编号，例如 `--gpu-ids "[0]"` 表示单卡，`--gpu-ids "[0,1]"` 表示双卡。旧的空格写法（`--gpu-ids 0 1`）仍然兼容。
- `--env.scene.num-envs`：并行环境数量；越大吞吐越高，但显存 / 内存占用也越高。
- `--agent.max-iterations`：最大训练迭代数。
  - 对 `Unitree-G1-AntiFall-Curriculum` 而言，它表示**每个 stage** 的最大迭代数。
  - 对单独 stage 任务而言，它表示该次运行的总迭代数。
- `--agent.save-interval`：checkpoint 保存间隔；训练过程中会按该间隔输出 `model_*.pt`，并同步导出 `policy.onnx`。

课程默认策略：
- 训练在单个顶层进程中按固定顺序推进 stage。
- Stage0 默认以稳定可控行走指标晋级。
- Stage1 ~ Stage4b 默认以恢复率 / 恢复延迟指标晋级。
- 若 stage 未达标但达到 `--agent.max-iterations`，则自动推进到下一 stage。
- 当前课程主线已经统一为平地 push-kick 梯度，因此后续 stage 不再依赖 rough / slip / trip 特有的 critic 观测切换。

训练输出目录：

```text
logs/rsl_rl/g1_antifall/<date_time>_<stage>/...
logs/rsl_rl/g1_antifall_curriculum/<date_time>_curriculum/...
```

> **迁移提示：** push-kick 主线重置之前产生的老 checkpoint / manifest 仍然可能能加载，
> 但它们对应的是旧的 rough / slip / trip 后期语义，而不是当前这套主线课程。

### 1.2 G1 抗摔任务回放（Play）

完成 `Unitree-G1-AntiFall-Curriculum` 训练后，**请使用 stage 子目录里的
checkpoint 回放**，不要直接拿顶层的
`logs/rsl_rl/g1_antifall_curriculum/<run>_curriculum/model_*.pt` 去 play。

原因是：

- `stages/<index>_<stage>/model_*.pt` 保存的是实际可加载的策略权重；
- 顶层 `model_*.pt` 主要用于记录 curriculum 进度元信息；
- 顶层 `policy.onnx` 适合部署导出，不是 `scripts/play_antifall.py` 的输入。

先找到最新一次 curriculum 训练目录：

```bash
LATEST_RUN=$(ls -dt logs/rsl_rl/g1_antifall_curriculum/*_curriculum | head -n1)
ls "$LATEST_RUN/stages"
```

如果训练已经完整跑完，通常直接回放最后一阶段 `Stage4b`：

```bash
CKPT=$(ls -t "$LATEST_RUN"/stages/05_stage4b/model_*.pt | head -n1)

python scripts/play_antifall.py \
  --task Unitree-G1-AntiFall-Stage4b \
  --checkpoint-file "$CKPT"
```

如果训练停在中间某个 stage，请改成对应的 stage 目录和任务名：

| stage 目录 | play 时使用的 task |
| --- | --- |
| `stages/00_stage0` | `Unitree-G1-AntiFall-Stage0` |
| `stages/01_stage1` | `Unitree-G1-AntiFall-Stage1` |
| `stages/02_stage2` | `Unitree-G1-AntiFall-Stage2` |
| `stages/03_stage3` | `Unitree-G1-AntiFall-Stage3` |
| `stages/04_stage4a` | `Unitree-G1-AntiFall-Stage4a` |
| `stages/05_stage4b` | `Unitree-G1-AntiFall-Stage4b` |

常用附加参数：

- `--num-envs 1`：只开一个可视化环境，便于观察动作；
- `--device cpu` 或 `--device cuda:0`：指定推理设备；
- `--video`：程序启动后立即开始录制，直到你关闭 play 才结束；
- `scripts/play_antifall.py` 强制使用 MuJoCo native viewer，因此需要图形显示环境
  （`DISPLAY` 或 `WAYLAND_DISPLAY`）。
- `scripts/play_antifall.py` 现在默认支持 **鼠标拖拽扰动**：
  在 MuJoCo native viewer 里直接点击 / 拖拽机器人，即可模拟手推 / 脚踢式扰动。
- native viewer 启动时会打印**当前推力上限（单位 N）**，并对连续拖拽施加与训练一致的
  impulse budget；同时 `Enter` 和 `Backspace` 都可以用来 reset。
- 视频会保存在 checkpoint 同级目录下的 `videos/play/`（例如
  `stages/05_stage4b/videos/play/rl-video-step-0.mp4`），即使用 `Ctrl+C`
  退出也会在退出时正常落盘。
- AntiFall 的 play 模式还会预留更大的 contact buffer，避免激烈拖拽时终端反复打印
  `broadphase overflow` 警告。

如果你更想走通用入口，也可以直接调用：

```bash
python scripts/play.py Unitree-G1-AntiFall-Stage4b \
  --checkpoint_file="$CKPT" \
  --num_envs=1
```

`scripts/play.py` 默认 `--viewer=auto`：有图形界面时优先使用 native viewer，
无图形界面时会自动切到 viser viewer。

任务分阶段语义、benchmark 说明和已知限制，请参阅
[`doc/g1_antifall.md`](doc/g1_antifall.md)。

### 1.3 G1 Parkour 回放（Play）

仓库现在提供独立的深度相机版 G1 Parkour 回放入口：

- 默认任务：`Unitree-G1-Parkour`
- 默认地形：更长的、参考 InstinctLab 的 MuJoCo 复杂地形序列
  （上楼梯、下楼梯、gap 近似、方块障碍、mesh-box stepping stones；gap 最大 0.40 m）
- 默认显示：MuJoCo native viewer + 实时 **policy 输入深度图**窗口
- 默认速度指令：`terrain-route`，会沿着任务里的
  `g1_parkour_route_waypoints` 生成速度指令，尽量让机器人按地形资产顺序走过去，
  而不是只给固定 x 方向速度导致越走越偏。可用 `--terrain-route-speed`
  显式指定路线跟随模式下的行走速度。
- 默认停止方式：按路线终点自动计算足够的运行时间，避免旧的短 debug 时长还没走完就停止。
- 可选视频录制：`--video` 默认录制 1080p MP4，并保存到导出模型文件同目录；
  可用 `--video-dir` 指定其他保存目录。

直接运行默认可视化：

```bash
python scripts/play_parkour.py
```

常用调试命令：

```bash
# 无窗口 / CI smoke 验证。
python scripts/play_parkour.py --validate-walk \
  --viewer none \
  --no-depth-viewer \
  --max-steps 20

# 关闭路线跟随，改用固定速度指令。
python scripts/play_parkour.py \
  --command-mode fixed \
  --command-x 0.25 \
  --command-y 0.0 \
  --command-yaw 0.0

# 路线跟随时显式指定行走速度，并保存视频到指定目录。
python scripts/play_parkour.py \
  --terrain-route-speed 0.35 \
  --video \
  --video-dir /tmp/parkour-videos
```

MuJoCo native viewer 和深度图窗口需要图形显示环境（`DISPLAY` 或
`WAYLAND_DISPLAY`）。如果只想做无头验证，请显式加
`--viewer none --no-depth-viewer`。

#### C++/DDS 手动运行

仓库不再在 `scripts/` 下保留 Python smoke harness。C++/DDS 验证请直接启动
simulator 和 controller，让诊断命令贴近实际部署二进制。simulator 默认加载和
`scripts/play_parkour.py` 对齐的确定性复杂 parkour XML：

```bash
# 建议先清理旧的 DDS 仿真/控制进程，避免 controller 连到旧 simulator。
pkill -f unitree_mujoco_parkour || true
pkill -f g1_parkour_ctrl || true

# 终端 1：启动 simulator。可视化模式会同时打开 MuJoCo 和深度窗口。
./simulate/build/unitree_mujoco_parkour --network lo

# 终端 2：等待 simulator ready 后启动交互 controller。
./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo
```

需要自动路线跟随时，直接给 controller 二进制传
`--sim-autostart-parkour`，并按需设置速度/深度相关参数。无头诊断时给
`unitree_mujoco_parkour` 传 `--headless --headless-seconds <N>`。如果只想隔离
控制链路而不依赖实时深度图，可给 controller 设置
`G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5`；如果要关闭 simulator 侧实时深度桥接，
用 `G1_PARKOUR_DEPTH_BRIDGE=0` 启动 simulator。

controller 终端里的键盘控制：

- `w` / `up`：只有按住时才按默认 `--sim-command-x`（`0.30 m/s`）向前走。
- `k`：如果你显式关闭默认交互模式并停在 `FixStand`，则用它进入 Parkour。
- `+` / `=` 和 `-`：按 policy keyboard step 调整按住 `w` 时的前进速度。
- `a` / `left` / `q`：左转；`d` / `right` / `e`：右转；`c`：停止转向。
- 松开移动键，或按 `s` / `down` / `x` / `space`：在 Parkour 中回到 idle-hold
  命令；`p`：Passive。

### 2. 动作模仿训练

训练 Unitree G1 模仿参考动作序列。

<div style="margin-left: 20px;">

#### 2.1 准备动作文件

将准备好的 csv 格式的动作文件保存在 mjlab/motions/g1/ 目录下，执行下面的指令将其转为训练可用的 npz 文件：

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50 \
--robot g1 # g1 or g1_23dof
```

**npz文件默认保存路径为**：`src/motions/g1/...`

#### 2.2 训练

确保有可用的npz文件之后，执行以下指令进行训练：

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

可用任务:
  - Unitree-G1-Tracking-No-State-Estimation
  - Unitree-G1-23Dof-Tracking-No-State-Estimation

</div>

> [!NOTE]
> 有关动作模仿训练的详细说明，请参阅BeyondMimic 文档
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  参数说明
- `--env.scene`: 仿真场景配置，包括环境数量（num_envs）、物理仿真步长、地面类型、重力、随机扰动等参数。
- `--env.observations`: 观测空间配置，控制训练时输入到策略网络的状态信息，如关节位置、速度、IMU等内容。
- `--env.rewards`: 奖励函数配置，定义每步训练时的优化目标。
- `--env.commands`: 控制命令配置，用于生成训练时随机或指定的速度 / 姿态 / 动作指令。
- `--env.terminations`: 终止条件配置，定义训练 episode 的结束条件。
- `--agent.seed`: 训练随机种子，用于结果复现，不同 seed 会导致策略略有差异。
- `--agent.resume`: 是否从上次中断的 checkpoint 继续训练。 设置为 True 时，会自动加载最近一次保存的 .pt 模型文件。
- `--agent.policy`: 策略网络结构配置，例如 MLP 层数、隐藏维度、激活函数等。
- `--agent.algorithm`: 强化学习算法配置。可设置优化超参数，如学习率、批量大小、GAE λ 等。

**默认保存训练结果**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. 仿真验证

如果想要在 MuJoCo 中查看训练效果，可以运行以下命令：

查看速度跟踪训练效果：
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

查看动作模仿训练效果：
```bash
python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**说明**：

- 训练时在每次保存模型时会同步导出 policy.onnx 文件在同层目录下，可用于实物部署。

**效果**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. 实物部署

实物部署前先确保主机安装了下列通信工具：
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 4.1 启动机器人
将机器人在吊装状态下启动，并等待机器人进入 `零力矩模式`

#### 4.2 进入调试模式
确保机器人处于 `零力矩模式` 的情况下，按下遥控器的 `L2+R2`组合键；此时机器人会进入`调试模式`, `调试模式`下机器人关节处于阻尼状态。

#### 4.3 连接机器人
使用网线连接电脑与机器人网口，并修改网络配置如下：
- 地址：`192.168.123.222`
- 子网掩码：`255.255.255.0`

然后使用 `ifconfig` 命令查看与机器人连接的网卡名称，记录后用于启动参数。

#### 4.4 编译
以 Unitree G1 速度控制为例（其他机器人同理）。
将策略文件（`policy.onnx`）放入`deploy/robots/g1/config/policy/velocity/vo/exported` 下，然后执行：

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

#### 4.5 部署

## 4.5.1 仿真部署

在实物部署前，建议使用[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)进行仿真部署，防止实物机器人出现异常动作。本框架已将其集成。

编译unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

启动仿真器(注意此处需连接上手柄才能启动)：

```bash
./simulate/build/unitree_mujoco
```

可在 `simulate/config` 中选择对应机器人

启动仿真控制程序：

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

## 4.5.2 实物部署

启动实物控制程序：

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

**参数说明**：
- `network`: 连接机器人网卡名称，仿真部署使用 `lo`，实物机器人如 `enp5s0`(可使用 `ifconfig` 指令查看)

</div>

**实物效果**：

| Go2                                                    | G1                                                    | H1_2                                                    | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [mjlab](https://github.com/mujocolab/mjlab.git): 构建训练与运行代码的基础。
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): 用于动作跟踪的通用人形机器人控制框架。
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): 提供 GPU 加速渲染与仿真接口。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
