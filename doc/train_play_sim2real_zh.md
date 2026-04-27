# Train → Play → Sim2Real 检查清单

本清单适用于仓库中的所有 policy lane。目标是在真实机器人之前发现 contract 和 runtime
不一致问题。

## 1. Train

1. 选择任务 id，并确认模块成熟度：
   - 基础速度任务和 GetUp 是可训练的 MJLab 任务。
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
   - AntiFall：`scripts/play_antifall.py`
   - GetUp：`scripts/play_getup.py`
   - Parkour：`scripts/play_parkour.py`
3. 检查 reset pose、命令方向、action scale、joint order 和 viewer 诊断。
4. Parkour 先用 `--check-contract` 检查深度 contract；无头短测使用
   `--validate-walk --viewer none --no-depth-viewer`。
5. 只有 Python play 行为符合预期后才进入下一阶段。

## 3. Simulator / C++ loopback

1. 构建 simulator 和 controller。
2. 本机仿真使用 loopback 网络（`--network lo`）。
3. 启动前清理旧 simulator/controller 进程，避免 DDS 连到旧进程。
4. 从低速度和保守模式开始。
5. 如果 C++ 行为和 Python play 不一致，优先比较：
   - joint order
   - action order
   - 默认姿态
   - startup blend
   - command frame
   - depth/camera validity

## 4. 真实机器人门槛

simulator 路径稳定前不要上硬件。电机使能前：

- 确认网卡和 DDS domain 正确。
- 确认 controller 源码版本与 policy contract 对应。
- 低速开始，安全员随时准备切 Passive。
- 修改 action order、默认姿态或 PD gain 后，首次测试应悬空或有支撑。
- 常量深度或 headless 诊断只能作为 ablation，不代表地形通过能力。

## 各模块 done criteria

| 模块 | Train 完成标准 | Play 完成标准 | Sim2Real 门槛 |
| --- | --- | --- | --- |
| 基础速度 | checkpoint 稳定，tracking 指标符合预期。 | Python play 中命令方向正确并能稳定行走。 | C++/DDS simulator 低速稳定。 |
| AntiFall | Curriculum/stage checkpoint 在训练评估中能恢复。 | Native viewer 鼠标拖拽扰动后能恢复。 | 只做保守仿真/硬件扰动测试。 |
| GetUp | 所选 terrain 的 checkpoint 收敛。 | 相同 terrain play 能重复起身。 | 硬件先从 ground terrain 开始，再测 platform/wall/slope。 |
| Parkour | 导出 artifact contract 完整。 | 深度 contract 和 route-following replay 通过。 | 先通过 simulator live-depth route，再考虑真实深度硬件测试。 |
