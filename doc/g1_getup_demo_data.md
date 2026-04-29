# G1 GetUp AMP Demo Data Fallback

This document describes the optional, simulation-only AMP fallback for flat-ground
G1 GetUp.  The default `Unitree-G1-GetUp` HoST-parity task remains no-demo and
should be tried first.  AMP is registered separately as `Unitree-G1-GetUp-AMP`.

## What this first pass does and does not prove

- Proves the local data contract, canonical 23DoF joint projection, discriminator
  reward wiring, one-iteration AMP/PPO smoke, and diagnostic JSON path.
- Does **not** claim long-run convergence or real-robot readiness.
- Does **not** enable platform/wall/slope AMP yet.
- Does **not** commit third-party motion data to git.

## Directory layout

Prepared demo data lives outside git:

```text
data/motions/g1_getup_amp/
  manifest.json
  source_gate.json
  motions/
    <standardized get-up clip>.npz
```

`data/motions/g1_getup_amp/` and `data/huggingface/` are gitignored.  Tests use
tiny fixture NPZ files under `tests/fixtures/g1_getup_amp/` only.

## Candidate public sources

These sources must still pass the local source gate before real training:

| Candidate | Why useful | First-pass caveat |
| --- | --- | --- |
| [`openhe/g1-retargeted-motions`](https://huggingface.co/datasets/openhe/g1-retargeted-motions) | Hugging Face card reports Unitree G1 23DoF, 30 FPS, 174 motion files, and includes falls/get-ups in LAFAN1 categories. | Files are `.pkl`, not this repo's standardized `.npz`; source/subset licenses still need recording. |
| [`fleaven/Retargeted_AMASS_for_robotics`](https://huggingface.co/datasets/fleaven/Retargeted_AMASS_for_robotics) | Provides Unitree G1 retargeted AMASS data and explicit joint order. | G1 layout is 29DoF with xyzw root quaternion, so it needs conversion to wxyz plus explicit 29→23 projection. AMASS subset licenses must be respected. |

Do not assume either source contains a clean get-up clip for this task.  The prep
script must emit `source_gate.json`; real training is allowed only when the gate
is `GO` and at least one license-clear get-up/fall-recovery clip is accepted.

## Source gate fields

`source_gate.json` records:

- `status`: `GO` or `STOP`.
- `stop_reasons`: unresolved source/license, no accepted clips, unsupported joint
  projection, etc.
- `source_url` and `source_revision`.
- `dataset_host_license`.
- `upstream_license_restrictions`.
- accepted/rejected sequence counts.

STOP means: keep using fixture smoke/tests only; do not report real-data training
readiness.

## Standardized NPZ schema

Each accepted clip is written under `motions/` with:

- `joint_pos`: `[T, 23]`, canonical active G1 23DoF joint order.
- `joint_vel`: `[T, 23]`, same order.
- `root_pos_w`: `[T, 3]`.
- `root_quat_w`: `[T, 4]`, `wxyz`.
- `amp_obs`: `[T, 53]`, concatenated root position, root quaternion, joint pos,
  joint vel.
- `joint_names`, `source_joint_names`, `fps`, `tags`, `source`, `license`, and
  JSON `projection` metadata.

Shape-only arrays are rejected.  Shuffled joints are reordered by name.  Known
29DoF extras (`waist_roll_joint`, `waist_pitch_joint`, wrist pitch/yaw joints)
are dropped only if every canonical 23DoF joint is present.

## Commands

### 1. Prepare/validate local data

For fixture smoke:

```bash
python scripts/prepare_g1_getup_amp_data.py \
  --input tests/fixtures/g1_getup_amp \
  --output /tmp/g1_getup_amp_fixture \
  --validate-only
```

For real downloaded data, include source metadata:

```bash
python scripts/prepare_g1_getup_amp_data.py \
  --input data/raw/g1_getup_source \
  --output data/motions/g1_getup_amp \
  --source-url https://huggingface.co/datasets/openhe/g1-retargeted-motions \
  --source-revision <commit-or-snapshot-id> \
  --source-license MIT \
  --upstream-license "record ACCAD/LAFAN1/original subset restrictions" \
  --require-go
```

### 2. One-iteration smoke training

```bash
python scripts/train_getup_amp.py \
  --demo-data-dir data/motions/g1_getup_amp \
  --max-iterations 1 \
  --num-envs 4 \
  --headless-smoke \
  -- --agent.num-steps-per-env=2
```

The wrapper maps short flags to real `scripts/train.py`/Tyro overrides and runs
`Unitree-G1-GetUp-AMP`.  The default `scripts/train_getup.py` remains the no-demo
HoST path.

### 3. Diagnostic JSON

```bash
python scripts/evaluate_getup_amp.py \
  --demo-data-dir data/motions/g1_getup_amp \
  --policy-mode random \
  --compare-no-demo \
  --max-steps 32 \
  --viewer none \
  --output /tmp/g1_getup_amp_eval.json
```

The JSON contains torso-height, upright-alignment, termination, run-mode, AMP
score/reward, and source-gate fields.  In this first pass it is a data-path and
short-rollout diagnostic, not a naturalness certificate.

## Advanced: raw human motion → G1 demos

If no public retargeted get-up clip passes validation:

1. Obtain raw motion under a license that allows your intended use.
2. Retarget to the active G1 model using an external IK/retargeting tool.
3. Export root position/quaternion and joint positions with explicit joint names.
4. Convert quaternions to `wxyz` and velocities to rad/s.
5. Smooth/filter obvious mocap spikes; remove clips with foot sliding, impossible
   root heights, or self-collision artifacts.
6. Run `scripts/prepare_g1_getup_amp_data.py --require-go` and inspect
   `manifest.json`/`source_gate.json` before training.

A full retarget optimizer is intentionally not implemented in this pass.

## Troubleshooting

- **No accepted clips:** names/tags may not indicate get-up/fall recovery, or the
  source genuinely lacks useful clips.
- **Wrong joint count/order:** include `joint_names`; do not rely on shape.
- **29DoF data rejected:** only known full-G1 extras can be dropped; missing
  canonical joints fail closed.
- **License STOP:** record dataset-host and upstream subset restrictions.
- **No AMP reward in logs:** ensure task is `Unitree-G1-GetUp-AMP`, not the
  default `Unitree-G1-GetUp`.
- **NaN/Inf data:** clean/filter the raw clip before preparation.
