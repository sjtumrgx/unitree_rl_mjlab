# Train → Play → Sim2Real 检查清单

本清单适用于仓库中的所有 policy lane。目标是在真实机器人之前发现 contract 和 runtime
不一致问题。

## 1. Train

1. 选择任务 id，并确认模块成熟度：
   - 基础速度任务和 AMP-Locomotion 是可训练的 MJLab 任务。
   - AntiFall 通过 stage/curriculum 任务训练。
   - Parkour 当前主要是 play/deploy-first 的已导出深度策略 lane。
2. 除非模块提供 wrapper，否则优先使用通用训练入口：

   ```bash
   python scripts/train.py <TaskId> --env.scene.num-envs=4096
   ```

3. 在实验记录中保留命令行覆盖项：device、terrain、env 数量、最大迭代数、checkpoint 选择标准。
4. 区分 artifact：
   - `model_*.pt`：Python play / resume checkpoint。
   - `policy.onnx` 或 `actor.onnx`：部署 artifact。
   - `deploy.yaml`：部署侧 contract，包含 joint order、action scale、sensor/depth 设置。

## 2. Play

进入 C++/DDS 前：

1. 在 Python 中回放同一个 checkpoint/artifact。
2. 有任务专用 play 脚本时优先使用：
   - AntiFall：`tools/play_antifall.py`
   - AMP-Locomotion：使用通用 `scripts/play.py Unitree-G1-AMP-Flat` 或 `Unitree-G1-AMP-Rough`
   - Parkour：`tools/play_parkour.py`
3. 检查 reset pose、命令方向、action scale、joint order 和 viewer 诊断。
4. Parkour 先用 `--check-contract` 检查深度 contract；无头短测使用
   `--validate-walk --viewer none --no-depth-viewer`。
5. 只有 Python play 行为符合预期后才进入下一阶段。

## 3. Simulator / C++ loopback

1. 先确认 C++/DDS 前置依赖：
   - Unitree SDK2 已安装到 `/opt/unitree_robotics`，并包含 `unitree_sdk2`、`ddsc`、
     `ddscxx` 头文件和库。
   - 系统包包含 `cmake`、`libyaml-cpp-dev`、`libboost-all-dev`、`libeigen3-dev`、
     `libspdlog-dev`、`libfmt-dev`、`zlib1g-dev`。
   - ONNX Runtime 使用仓库内 `deploy/thirdparty/onnxruntime-linux-*-1.22.0/`，不用
     另装系统包。
2. 构建共用 simulator：

   ```bash
   cmake -S simulate -B simulate/build
   cmake --build simulate/build -j4
   ```

3. 为当前要验证的任务构建对应 controller。
4. 本机仿真使用 loopback 网络（`--network=lo`），真实机器人使用实际网卡名
   （例如 `eth0`、`enp*`，以 `ip addr` 输出为准）。
5. 启动前清理旧 simulator/controller 进程，避免 DDS 连到旧进程：

   ```bash
   pkill -f unitree_mujoco || true
   pkill -f g1_ctrl || true
   pkill -f g1_antifall_ctrl || true
   pkill -f g1_parkour_ctrl || true
   ```

6. 两个终端启动：先 simulator，等 MuJoCo/DDS bridge 起起来，再启动 controller。
7. 从低速度和保守模式开始。
8. 如果 C++ 行为和 Python play 不一致，优先比较：
   - joint order
   - action order
   - 默认姿态
   - startup blend
   - command frame
   - depth/camera validity

### 各任务命令矩阵

| 任务 | Controller 构建 | Simulator 终端 | Controller 终端 | 控制切换 | 真实机器人命令形态 |
| --- | --- | --- | --- | --- | --- |
| Velocity / 基础 G1 | `cmake -S deploy/robots/g1 -B deploy/robots/g1/build`<br>`cmake --build deploy/robots/g1/build -j4` | `./simulate/build/unitree_mujoco --network lo` | `./deploy/robots/g1/build/g1_ctrl --network=lo --keyboard` | 键盘 `f` 进 FixStand，`v` 进 Velocity，`w/s/a/d/q/e` 速度控制，松开停止，`p` 回 Passive；遥控器 `L2+Up` → `R2+A`，`L2+B` 回 Passive | `./deploy/robots/g1/build/g1_ctrl --network=<robot_nic> --keyboard` |
| AntiFall | `cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build`<br>`cmake --build deploy/robots/g1_antifall/build -j4` | `./simulate/build/unitree_mujoco --network lo` | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo --keyboard` | 键盘 `f` 进 FixStand，`v` 进 AntiFall，`w/s/a/d/q/e` 速度控制，`p` 回 Passive；遥控器 `L2+Up` → `R2+A`，`L2+B` 回 Passive | `./deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=<robot_nic> --keyboard` |
| AMP-Locomotion | 本次迁移只引入 Python 训练/play lane，暂不新增 C++ controller | `./simulate/build/unitree_mujoco --network lo` | Python play：`python scripts/play.py Unitree-G1-AMP-Flat --checkpoint-file <model.pt>` | 先完成训练/play；AMP runner 会导出 `policy.onnx` | 上硬件前需要补独立 C++ controller contract。 |
| Parkour | `cmake -S deploy/robots/g1_parkour -B deploy/robots/g1_parkour/build`<br>`cmake --build deploy/robots/g1_parkour/build -j4` | `./simulate/build/unitree_mujoco_parkour --network lo` | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo`<br>自动进入：`./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=lo --sim-autostart-parkour` | loopback 默认进入 Parkour idle-hold；按住 `w` / `up` 沿路线前进，松开停止，`+/-` 调速度，`a/d/q/e` 转向，`s/down/x/space` 回 idle，`p` 回 Passive；真实遥控器 `L2+Up` → `R2+X` | `./deploy/robots/g1_parkour/build/g1_parkour_ctrl --network=<robot_nic> --keyboard` |

无 GUI loopback 可在 simulator 命令上加 `--headless --headless-seconds <N>`。
Parkour 还支持在 simulator 侧设置 `G1_PARKOUR_DEPTH_BRIDGE=0`，在 controller 侧使用
`G1_PARKOUR_DEBUG_CONSTANT_DEPTH=0.5` 或 `--constant-depth <value>` 做深度 ablation。

### 启动后的通用操作顺序

1. Simulator 终端出现 MuJoCo 窗口或 headless 日志后，不要先按 controller 按键。
2. Controller 终端应打印 `Waiting for connection to robot...`，随后打印
   `Connected to robot.`；如果提示 lowcmd channel 被占用，先关掉旧 controller。
3. 键盘模式需要 controller 终端保持焦点；`--keyboard` 在非交互终端会直接报错。
4. 进入 RL 状态前先通过 FixStand，让机器人从 Passive 平滑站到默认姿态。
5. 任何异常抖动、姿态错误、深度异常或接触异常，先按 `p` 或遥控器 `L2+B` 回
   Passive，再停 controller。

## 4. 真实机器人门槛

simulator 路径稳定前不要上硬件。电机使能前：

- 确认网卡和 DDS domain 正确；真实机器人命令不要使用 `--network=lo`。
- 确认 controller 源码版本与 policy contract 对应。
- 低速开始，安全员随时准备切 Passive。
- 修改 action order、默认姿态或 PD gain 后，首次测试应悬空或有支撑。
- 常量深度或 headless 诊断只能作为 ablation，不代表地形通过能力。

## 各模块 done criteria

| 模块 | Train 完成标准 | Play 完成标准 | Sim2Real 门槛 |
| --- | --- | --- | --- |
| 基础速度 | checkpoint 稳定，tracking 指标符合预期。 | Python play 中命令方向正确并能稳定行走。 | C++/DDS simulator 低速稳定。 |
| AntiFall | Curriculum/stage checkpoint 在训练评估中能恢复。 | Native viewer 鼠标拖拽扰动后能恢复。 | 只做保守仿真/硬件扰动测试。 |
| AMP-Locomotion | 4 卡训练达到目标迭代数，并出现 recovery 指标。 | Python play 能回放统一 locomotion/recovery checkpoint。 | 上硬件前先补专用 C++ controller contract。 |
| Parkour | 导出 artifact contract 完整。 | 深度 contract 和 route-following replay 通过。 | 先通过 simulator live-depth route，再考虑真实深度硬件测试。 |
