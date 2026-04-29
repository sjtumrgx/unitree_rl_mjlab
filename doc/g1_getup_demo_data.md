# G1 GetUp AMP Demonstration Data Guide

This guide is for the optional **ground-only** AMP fallback task
`Unitree-G1-GetUp-AMP`.  The default `Unitree-G1-GetUp` HoST-parity task remains
no-demo; use AMP only when the default get-up policy is not natural enough.

The repository does not include large motion data.  Download data outside git,
prepare it into the local AMP schema, then train/play from the prepared directory.

## Data source

Use this dataset first:

- Hugging Face: <https://huggingface.co/datasets/openhe/g1-retargeted-motions>
- Local raw-data target: `data/raw/g1_getup_source/openhe/g1-retargeted-motions/`
- License metadata to record in `source_gate.json`: dataset card license `MIT`,
  plus original source restrictions for ACCAD/LAFAN1/DanceDB/etc.

Why this source: the dataset card describes Unitree G1 retargeted motions in
Python pickle format, 23 DoF, 30 FPS, and includes transition/fall/get-up-like
clips.  The prep tool in this repo accepts those `.pkl` files directly and
converts accepted get-up/fall-recovery candidates to the AMP `.npz` schema.

## Download

Install Hugging Face tooling if needed:

```bash
python -m pip install -U huggingface_hub
```

Download the dataset with the official Hugging Face CLI:

```bash
huggingface-cli download openhe/g1-retargeted-motions \
  --repo-type dataset \
  --local-dir data/raw/g1_getup_source/openhe/g1-retargeted-motions \
  --local-dir-use-symlinks False
```

Alternative with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions \
  data/raw/g1_getup_source/openhe/g1-retargeted-motions
```

Raw data under `data/raw/g1_getup_source/` is gitignored.  Do not move downloaded
motions into tracked source directories.

## Folder tree

After download:

```text
data/raw/g1_getup_source/
  openhe/
    g1-retargeted-motions/
      README.md
      ACCAD_retargeted/
        A10-_Lie_to_crouch_stageii.pkl
        A10_-_lie_to_crouch_stageii.pkl
        ...
      LAFAN1_retargeted/
        ...
      dance_db_retargeted/
        ...
      kungfu_retargeted/
        ...
```

After preparation:

```text
data/motions/g1_getup_amp/
  manifest.json
  source_gate.json
  motions/
    A10-_Lie_to_crouch_stageii.npz
    A10_-_lie_to_crouch_stageii.npz
    ...
```

After training:

```text
logs/rsl_rl/g1_getup_amp/
  <timestamp>_ground_amp/
    model_*.pt
    policy.onnx
    params/
      agent.yaml
      env.yaml
```

For C++/DDS deployment, the GetUp controller expects:

```text
deploy/robots/g1_getup/config/policy/getup/v0/
  exported/
    policy.onnx
```

## Prepare downloaded data

The prep step scans `.pkl` and `.npz` files, accepts get-up/fall-recovery-like
clips, projects them to the active G1 23DoF joint order, and writes
`manifest.json` plus `source_gate.json`.

Use a real dataset revision.  For a Git clone, obtain it with:

```bash
git -C data/raw/g1_getup_source/openhe/g1-retargeted-motions rev-parse HEAD
```

Prepare data:

```bash
python scripts/prepare_g1_getup_amp_data.py \
  --input data/raw/g1_getup_source/openhe/g1-retargeted-motions \
  --output data/motions/g1_getup_amp \
  --source-url https://huggingface.co/datasets/openhe/g1-retargeted-motions \
  --source-revision <dataset-commit-or-snapshot-id> \
  --source-license MIT \
  --upstream-license "ACCAD/LAFAN1/DanceDB/etc. original source restrictions reviewed" \
  --require-go
```

Training is blocked unless `data/motions/g1_getup_amp/source_gate.json` exists
and has `"status": "GO"`.

## Prepared data schema

Each accepted `.npz` contains:

```text
joint_pos              [T, 23] canonical active G1 23DoF order
joint_vel              [T, 23]
root_pos_w             [T, 3]
root_quat_w            [T, 4] wxyz
amp_obs                [T, 53] root pos + root quat + joint pos + joint vel
joint_names            [23]
source_joint_names     [23]
fps, tags, source, license, projection
```

The canonical joint order is derived from
`src/assets/robots/unitree_g1/xmls/g1_23dof.xml`.  Shape-only data is rejected
unless it is handled by an explicit source adapter such as the OpenHE `.pkl`
adapter, which records that source-format assumption in the `projection` field.

## Train

Run formal AMP training from the prepared data directory:

```bash
python scripts/train_getup_amp.py \
  --demo-data-dir data/motions/g1_getup_amp \
  --num-envs 4096 \
  --max-iterations 10001
```

Equivalent generic entrypoint:

```bash
python scripts/train.py Unitree-G1-GetUp-AMP \
  --agent.algorithm.demo-data-dir=data/motions/g1_getup_amp \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001
```

The default no-demo terrain task remains:

```bash
python scripts/train_getup.py --terrain ground -- --env.scene.num-envs=4096
```

## Play

Replay the AMP checkpoint in Python/MuJoCo:

```bash
python scripts/play.py Unitree-G1-GetUp-AMP \
  --checkpoint_file logs/rsl_rl/g1_getup_amp/<run>/model_*.pt \
  --num_envs 1 \
  --viewer native
```

Use `--viewer viser` for a browser viewer or `--video` if you want a recorded
video in the run directory.  Keep `data/motions/g1_getup_amp/source_gate.json`
available because the AMP algorithm config validates the demo-data gate when the
runner is constructed.

## Sim2Real / Unitree simulator path

AMP is still a fallback policy and must pass the same simulator safety gates as
the no-demo GetUp policy.  Do not put it on real hardware directly after
training.

1. Confirm Python play recovers from the intended ground fallen poses.
2. Copy or symlink the exported policy into the GetUp deployment bundle:

   ```bash
   mkdir -p deploy/robots/g1_getup/config/policy/getup/v0/exported
   cp logs/rsl_rl/g1_getup_amp/<run>/policy.onnx \
     deploy/robots/g1_getup/config/policy/getup/v0/exported/policy.onnx
   ```

3. Build the controller:

   ```bash
   cmake -S deploy/robots/g1_getup -B deploy/robots/g1_getup/build
   cmake --build deploy/robots/g1_getup/build -j4
   ```

4. Start Unitree MuJoCo simulator and then the controller on loopback:

   ```bash
   ./simulate/build/unitree_mujoco
   ./deploy/robots/g1_getup/build/g1_getup_ctrl --network=lo --keyboard
   ```

5. Use keyboard `f` then `g` (`Passive -> FixStand -> GetUp`) only after the
   simulator robot is in the expected supported start state.

6. For real robot testing, keep the robot physically supported, keep an operator
   ready to switch to Passive, and verify action order/PD gains/torque limits
   against the exported policy metadata first.

## Troubleshooting

- **`source_gate.json` is STOP:** review source URL, revision, dataset license,
  upstream license restrictions, and accepted clip count.
- **No accepted clips:** inspect filenames under `ACCAD_retargeted/` and
  `LAFAN1_retargeted/`; get-up-like names such as `Lie_to_crouch` are expected
  to pass before generic locomotion clips.
- **Training fails before env creation:** the source gate is missing or not GO.
- **Policy looks unnatural:** curate fewer/higher-quality get-up clips before
  increasing reward scale; do not enable platform/wall/slope AMP until ground
  succeeds.
- **Sim2Real mismatch:** check exported `policy.onnx`, joint order, action scale,
  default pose, PD gains, and initial fallen geometry before tuning rewards.
