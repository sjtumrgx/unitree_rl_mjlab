# G1 GetUp demo data workflow

_最短路径：本地 `.pkl` 动作数据 -> 转成训练可用 `.npz` -> 用 G1 MuJoCo 模型播放检查 -> 选择这些 `.npz` 开始 GetUp AMP 训练。_

---

## 📁 Folder tree

本文假设原始动作数据已经放在本地 `data/g1-retargeted-motions/`。不在这里重复下载步骤；只关心下载完成后怎么选、转、看、训。

```text
unitree_rl_mjlab/
  data/
    g1_getup_amp.yaml                 # 本流程默认读取的配置
    g1-retargeted-motions/            # 本地原始数据，不进 git
      README.md
      lafan1_retargeted/
        fallAndGetUp1_subject1.pkl
        fallAndGetUp1_subject4.pkl
        fallAndGetUp1_subject5.pkl
        fallAndGetUp2_subject2.pkl
        fallAndGetUp2_subject3.pkl
        fallAndGetUp3_subject1.pkl
      ACCAD_retargeted/
        ...
    motions/
      g1_getup_amp/                   # prepare 后生成，不进 git
        manifest.json
        source_gate.json
        motions/
          fallAndGetUp1_subject1.npz
          fallAndGetUp1_subject4.npz
          ...
  scripts/
    prepare_g1_getup_amp_data.py      # pkl/npz -> 标准 AMP npz
    play_g1_getup_amp_data.py         # 用 G1 MuJoCo 模型播放 npz
    train_getup_amp.py                # 用配置里的 npz 数据训练 AMP
  src/assets/robots/unitree_g1/xmls/
    g1_23dof.xml                      # 播放动作时使用的 G1 模型
```

## ⚙️ Edit one YAML file

默认配置在 `data/g1_getup_amp.yaml`。通常只改三处：

```yaml
prepare:
  output_dir: data/motions/g1_getup_amp
  inputs:
    # 可以写文件夹：脚本会递归扫描下面的 .pkl
    - data/g1-retargeted-motions/lafan1_retargeted

    # 也可以只写你确定要用的若干个 .pkl
    # - data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp1_subject1.pkl
    # - data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp2_subject2.pkl

play:
  xml: src/assets/robots/unitree_g1/xmls/g1_23dof.xml
  npz_files:
    - data/motions/g1_getup_amp/motions/fallAndGetUp1_subject1.npz
  speed: 1.0
  loop: false

train:
  demo_data_dir: data/motions/g1_getup_amp
  npz_files:
    - data/motions/g1_getup_amp/motions/fallAndGetUp1_subject1.npz
    - data/motions/g1_getup_amp/motions/fallAndGetUp2_subject2.npz
  num_envs: 4096
  max_iterations: 10001
```

`dataset:` 里的来源信息只用于写入本地 `source_gate.json`。一般不用碰；如果你想记录具体快照，把 `dataset_commit_or_snapshot_id` 改成自己的数据集 commit 或 snapshot id 即可。

## 🔄 Prepare selected local data

把 YAML 里 `prepare.inputs` 指定的动作转成训练可用的 `.npz`：

```bash
python scripts/prepare_g1_getup_amp_data.py
```

脚本会：

- 读取 `data/g1_getup_amp.yaml`
- 对文件夹递归扫描 `.pkl`，对文件路径只处理该文件
- 把 OpenHE/G1 retargeted `.pkl` 转成标准 AMP `.npz`
- 写入 `data/motions/g1_getup_amp/manifest.json`
- 写入 `data/motions/g1_getup_amp/motions/*.npz`

临时想覆盖 YAML，也可以用 CLI 指一个文件夹或文件：

```bash
python scripts/prepare_g1_getup_amp_data.py \
  --input data/g1-retargeted-motions/lafan1_retargeted \
  --output data/motions/g1_getup_amp
```

## 👀 Play converted data on G1

先在 YAML 的 `play.npz_files` 里选择要看的 `.npz`，然后运行：

```bash
python scripts/play_g1_getup_amp_data.py
```

这会打开 MuJoCo viewer，并使用 `src/assets/robots/unitree_g1/xmls/g1_23dof.xml` 把动作重定向到 G1 23DoF 模型上播放。

只想做无窗口检查：

```bash
python scripts/play_g1_getup_amp_data.py --validate-only
```

临时播放某个 `.npz`，不改 YAML：

```bash
python scripts/play_g1_getup_amp_data.py \
  --npz-file data/motions/g1_getup_amp/motions/fallAndGetUp1_subject1.npz \
  --speed 1.0
```

播放多个动作：

```bash
python scripts/play_g1_getup_amp_data.py \
  --npz-file data/motions/g1_getup_amp/motions/fallAndGetUp1_subject1.npz \
  --npz-file data/motions/g1_getup_amp/motions/fallAndGetUp2_subject2.npz \
  --play-all
```

如果动作看起来不自然，回到 `data/g1_getup_amp.yaml` 调整 `prepare.inputs`，只保留质量更好的 get-up/fall-recovery 片段，再重新 prepare 和 play。

## 🏋️ Train with selected `.npz`

确认播放没问题后，把要训练的 `.npz` 写到 YAML 的 `train.npz_files`，然后直接开始训练：

```bash
python scripts/train_getup_amp.py
```

如果 `train.npz_files` 存在，脚本会在 `train.demo_data_dir` 下生成 `selected_manifest.json`，并只把这些 `.npz` 转发给 `Unitree-G1-GetUp-AMP`。

等价于显式写：

```bash
python scripts/train_getup_amp.py \
  --demo-data-dir data/motions/g1_getup_amp \
  --num-envs 4096 \
  --max-iterations 10001
```

训练任务是 `Unitree-G1-GetUp-AMP`，输出目录类似：

```text
logs/rsl_rl/g1_getup_amp/
  <timestamp>_ground_amp/
    model_*.pt
    policy.onnx
    params/
      agent.yaml
      env.yaml
```

默认无 demo 的 GetUp 训练仍然是：

```bash
python scripts/train_getup.py --terrain ground -- --env.scene.num-envs=4096
```

## ✅ Recommended loop

1. 在 `data/g1_getup_amp.yaml` 的 `prepare.inputs` 里选动作
2. 运行 `python scripts/prepare_g1_getup_amp_data.py`
3. 在 `play.npz_files` 里选一个或多个生成的 `.npz`
4. 运行 `python scripts/play_g1_getup_amp_data.py`，用 MuJoCo 观察动作
5. 把确认过的 `.npz` 写到 `train.npz_files`
6. 运行 `python scripts/train_getup_amp.py`

## 🧯 Troubleshooting

- **找不到 `.pkl`：** 检查 `prepare.inputs` 路径是否相对 repo 根目录，或改成绝对路径
- **没有生成 `.npz`：** 文件名需要像 get-up/fall/recovery 相关动作；普通 walk/run 会被拒绝
- **播放窗口打不开：** 先用 `--validate-only` 确认数据和 G1 XML 能被加载，再检查本机 MuJoCo viewer 环境
- **训练启动前失败：** 先重新运行 prepare，确认 `data/motions/g1_getup_amp/manifest.json` 和 `source_gate.json` 都存在
- **动作不适合训练：** 不要先调奖励；先缩小 `prepare.inputs`，只保留播放效果好的 `.npz`
