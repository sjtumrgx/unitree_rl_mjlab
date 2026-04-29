# G1 GetUp AMP Demonstration Data Guide

This guide is for the optional **ground-only** AMP fallback task
`Unitree-G1-GetUp-AMP`.  The default `Unitree-G1-GetUp` HoST-parity task remains
no-demo; use AMP only when the default get-up policy is not natural enough.

Large motion files are not tracked in git.  The real local raw-data path for
this workspace is:

```text
~/unitree_rl_mjlab/data/g1-retargeted-motions
```

Validate/replay the selected clips from that path, prepare them into the local
AMP manifest directory, then point AMP training at the prepared directory.

## Data source

Use this public dataset first:

- Hugging Face: <https://huggingface.co/datasets/openhe/g1-retargeted-motions>
- Local raw-data target used by the tools:
  `~/unitree_rl_mjlab/data/g1-retargeted-motions/`
- Prepared AMP-data target:
  `~/unitree_rl_mjlab/data/motions/g1_getup_amp/`
- Dataset-host license metadata for `source_gate.json`: `MIT`
- Upstream restrictions to review: LAFAN1 original data restrictions for the
  selected `lafan1_retargeted` clips.

The selected first-pass GetUp subset is:

```text
~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/
  fallAndGetUp1_subject1.pkl
  fallAndGetUp1_subject4.pkl
  fallAndGetUp1_subject5.pkl
  fallAndGetUp2_subject2.pkl
  fallAndGetUp2_subject3.pkl
  fallAndGetUp3_subject1.pkl
```

These are the defaults in `scripts/play_g1_getup_amp_data.py`.  Use repeated
`--motion-file <path>` arguments if you want a different curated subset.

## Download

Install Hugging Face tooling if needed:

```bash
python -m pip install -U huggingface_hub
```

Download the dataset with the official Hugging Face CLI:

```bash
huggingface-cli download openhe/g1-retargeted-motions \
  --repo-type dataset \
  --local-dir ~/unitree_rl_mjlab/data/g1-retargeted-motions \
  --local-dir-use-symlinks False
```

Alternative with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions \
  ~/unitree_rl_mjlab/data/g1-retargeted-motions
```

Raw data under `~/unitree_rl_mjlab/data/g1-retargeted-motions/` is gitignored.
Do not move third-party motion files into tracked source directories.

For a Git clone, record the dataset revision with:

```bash
git -C ~/unitree_rl_mjlab/data/g1-retargeted-motions rev-parse HEAD
```

For a Hugging Face snapshot download, use the snapshot commit shown by the
Hugging Face web UI or your local download metadata.

## Folder tree

After download:

```text
~/unitree_rl_mjlab/data/g1-retargeted-motions/
  README.md
  accad_retargeted/...
  lafan1_retargeted/
    fallAndGetUp1_subject1.pkl
    fallAndGetUp1_subject4.pkl
    fallAndGetUp1_subject5.pkl
    fallAndGetUp2_subject2.pkl
    fallAndGetUp2_subject3.pkl
    fallAndGetUp3_subject1.pkl
  ...
```

After selected-clip preparation:

```text
~/unitree_rl_mjlab/data/motions/g1_getup_amp/
  manifest.json
  source_gate.json
  motions/
    fallAndGetUp1_subject1.npz
    fallAndGetUp1_subject4.npz
    fallAndGetUp1_subject5.npz
    fallAndGetUp2_subject2.npz
    fallAndGetUp2_subject3.npz
    fallAndGetUp3_subject1.npz
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

## Prepare all local source data

`scripts/prepare_g1_getup_amp_data.py` now defaults to the real local raw-data
path and prepared-data path:

```bash
python scripts/prepare_g1_getup_amp_data.py \
  --source-revision <dataset-commit-or-snapshot-id> \
  --require-go
```

Equivalent explicit form:

```bash
python scripts/prepare_g1_getup_amp_data.py \
  --input ~/unitree_rl_mjlab/data/g1-retargeted-motions \
  --output ~/unitree_rl_mjlab/data/motions/g1_getup_amp \
  --source-url https://huggingface.co/datasets/openhe/g1-retargeted-motions \
  --source-revision <dataset-commit-or-snapshot-id> \
  --source-license MIT \
  --upstream-license "LAFAN1 original source restrictions reviewed" \
  --require-go
```

This broad prepare command scans `.pkl` and `.npz` files under the input path.
If you want to prepare only the six curated `fallAndGetUp` clips, use the
playback helper below.

## Validate and replay the selected data

Validate the six default LAFAN1 `fallAndGetUp` clips, convert OpenHE quaternion
`xyzw` to MuJoCo/AMP `wxyz`, write the AMP manifest, and run a headless MuJoCo
kinematic check:

```bash
python scripts/play_g1_getup_amp_data.py \
  --source-revision <dataset-commit-or-snapshot-id> \
  --require-go \
  --validate-only
```

The command writes
`~/unitree_rl_mjlab/data/motions/g1_getup_amp/manifest.json`,
`~/unitree_rl_mjlab/data/motions/g1_getup_amp/source_gate.json`, and one
prepared `.npz` per accepted clip under
`~/unitree_rl_mjlab/data/motions/g1_getup_amp/motions/`.

Replay the first accepted clip in the native MuJoCo viewer:

```bash
python scripts/play_g1_getup_amp_data.py \
  --source-revision <dataset-commit-or-snapshot-id> \
  --motion-index 0 \
  --speed 1.0
```

Replay every accepted clip sequentially:

```bash
python scripts/play_g1_getup_amp_data.py \
  --source-revision <dataset-commit-or-snapshot-id> \
  --play-all
```

Prepare/play a custom subset:

```bash
python scripts/play_g1_getup_amp_data.py \
  --motion-file ~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp1_subject1.pkl \
  --motion-file ~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/fallAndGetUp2_subject2.pkl \
  --source-revision <dataset-commit-or-snapshot-id> \
  --require-go \
  --validate-only
```

## Prepared data schema

Each accepted `.npz` contains:

```text
joint_pos              [T, 23] canonical active G1 23DoF order
joint_vel              [T, 23]
root_pos_w             [T, 3]
root_quat_w            [T, 4] wxyz; converted from OpenHE xyzw
amp_obs                [T, 53] root pos + root quat + joint pos + joint vel
joint_names            [23]
source_joint_names     [23]
fps, tags, source, license, projection
```

The canonical joint order is derived from
`src/assets/robots/unitree_g1/xmls/g1_23dof.xml`.  Shape-only data is rejected
unless it is handled by an explicit source adapter such as the OpenHE `.pkl`
adapter, which records that source-format assumption in the `projection` field.

## Train with the prepared data

Run formal AMP training from the prepared selected-clip directory:

```bash
python scripts/train_getup_amp.py \
  --demo-data-dir ~/unitree_rl_mjlab/data/motions/g1_getup_amp \
  --num-envs 4096 \
  --max-iterations 10001
```

Equivalent generic entrypoint:

```bash
python scripts/train.py Unitree-G1-GetUp-AMP \
  --agent.algorithm.demo-data-dir=$HOME/unitree_rl_mjlab/data/motions/g1_getup_amp \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001
```

The default no-demo terrain task remains:

```bash
python scripts/train_getup.py --terrain ground -- --env.scene.num-envs=4096
```

`Unitree-G1-GetUp-AMP` refuses to train unless the selected data directory has a
`source_gate.json` with `"status": "GO"`.

## Play the trained policy

Replay the AMP checkpoint in Python/MuJoCo:

```bash
python scripts/play.py Unitree-G1-GetUp-AMP \
  --checkpoint_file logs/rsl_rl/g1_getup_amp/<run>/model_*.pt \
  --num_envs 1 \
  --viewer native
```

Use `--viewer viser` for a browser viewer or `--video` if you want a recorded
video in the run directory.  Keep
`~/unitree_rl_mjlab/data/motions/g1_getup_amp/source_gate.json` available
because the AMP algorithm config validates the demo-data gate when the runner is
constructed.

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

- **`source_gate.json` is STOP:** pass a real `--source-revision`, review source
  URL, dataset license, upstream restrictions, and accepted clip count.
- **Missing `joblib`:** install this repo with `pip install -e .` or run
  `python -m pip install joblib`; OpenHE `.pkl` files use joblib array wrappers.
- **No accepted clips:** confirm filenames include get-up/fall-recovery content;
  the selected `fallAndGetUp*.pkl` files above are expected to pass.
- **Training fails before env creation:** the source gate is missing or not GO.
- **Policy looks unnatural:** curate fewer/higher-quality get-up clips before
  increasing reward scale; do not enable platform/wall/slope AMP until ground
  succeeds.
- **Sim2Real mismatch:** check exported `policy.onnx`, joint order, action scale,
  default pose, PD gains, and initial fallen geometry before tuning rewards.
