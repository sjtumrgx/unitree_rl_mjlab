# G1 GetUp: Train → Play → Sim2Real Notes

G1 GetUp ports HoST-style get-up terrain variants into MJLab while keeping the
normal Unitree RL MJLab workflow.  The default task is no-demo HoST parity;
`Unitree-G1-GetUp-AMP` is a separate ground-only fallback that uses prepared
retargeted demonstration data.

## Task map

| Path | MJLab task | Scope |
| --- | --- | --- |
| Default ground | `Unitree-G1-GetUp` + `--getup-terrain=ground` | no-demo HoST ground get-up |
| Default platform | `Unitree-G1-GetUp` + `--getup-terrain=platform` | no-demo HoST platform variant |
| Default wall | `Unitree-G1-GetUp` + `--getup-terrain=wall` | no-demo HoST wall variant |
| Default slope | `Unitree-G1-GetUp` + `--getup-terrain=slope` | no-demo HoST slope variant |
| AMP fallback | `Unitree-G1-GetUp-AMP` | ground-only demo-data fallback |

Core code lives in:

- `src/tasks/velocity/config/g1_getup/env_cfgs.py`
- `src/tasks/velocity/config/g1_getup/rl_cfg.py`
- `src/tasks/velocity/mdp/getup/`
- `src/tasks/velocity/rl/getup_amp.py`
- `src/tasks/velocity/rl/getup_amp_data.py`

## Default no-demo training

Convenience wrapper:

```bash
python scripts/train_getup.py --terrain ground -- --env.scene.num-envs=4096
python scripts/train_getup.py --terrain platform -- --env.scene.num-envs=4096
python scripts/train_getup.py --terrain wall -- --env.scene.num-envs=4096
python scripts/train_getup.py --terrain slope -- --env.scene.num-envs=4096
```

Generic form:

```bash
python scripts/train.py Unitree-G1-GetUp --getup-terrain=platform
```

The terrain flag affects terrain mix, reset state, assist force, and run naming.
Use separate runs for each terrain unless you intentionally test cross-terrain
transfer.

## AMP fallback data setup

Use the AMP fallback only after the default ground task is insufficient.  The
recommended public source is:

```text
https://huggingface.co/datasets/openhe/g1-retargeted-motions
```

Download to this ignored raw-data directory:

```bash
huggingface-cli download openhe/g1-retargeted-motions \
  --repo-type dataset \
  --local-dir ~/unitree_rl_mjlab/data/g1-retargeted-motions \
  --local-dir-use-symlinks False
```

The first-pass curated subset is the six LAFAN1 clips used by
`scripts/play_g1_getup_amp_data.py` by default:

```text
~/unitree_rl_mjlab/data/g1-retargeted-motions/lafan1_retargeted/
  fallAndGetUp1_subject1.pkl
  fallAndGetUp1_subject4.pkl
  fallAndGetUp1_subject5.pkl
  fallAndGetUp2_subject2.pkl
  fallAndGetUp2_subject3.pkl
  fallAndGetUp3_subject1.pkl
```

Validate/replay those clips and prepare the exact training manifest:

```bash
python scripts/play_g1_getup_amp_data.py \
  --source-revision <dataset-commit-or-snapshot-id> \
  --require-go \
  --validate-only
```

Expected tree:

```text
~/unitree_rl_mjlab/data/g1-retargeted-motions/
  lafan1_retargeted/*.pkl

~/unitree_rl_mjlab/data/motions/g1_getup_amp/
  manifest.json
  source_gate.json
  motions/fallAndGetUp*.npz
```

See `doc/g1_getup_demo_data.md` for full source-gate, schema, and licensing
details.

## AMP fallback training

Formal AMP training:

```bash
python scripts/train_getup_amp.py \
  --demo-data-dir ~/unitree_rl_mjlab/data/motions/g1_getup_amp \
  --num-envs 4096 \
  --max-iterations 10001
```

Equivalent generic form:

```bash
python scripts/train.py Unitree-G1-GetUp-AMP \
  --agent.algorithm.demo-data-dir=$HOME/unitree_rl_mjlab/data/motions/g1_getup_amp \
  --env.scene.num-envs=4096 \
  --agent.max-iterations=10001
```

`Unitree-G1-GetUp-AMP` refuses to train unless
`~/unitree_rl_mjlab/data/motions/g1_getup_amp/source_gate.json` exists and is `GO`.

## Play

Default no-demo terrain play:

```bash
python scripts/play_getup.py --terrain ground -- \
  --checkpoint_file logs/rsl_rl/g1_getup/<run>/model_*.pt
```

AMP fallback play:

```bash
python scripts/play.py Unitree-G1-GetUp-AMP \
  --checkpoint_file logs/rsl_rl/g1_getup_amp/<run>/model_*.pt \
  --num_envs 1 \
  --viewer native
```

Demonstration data playback before training:

```bash
python scripts/play_g1_getup_amp_data.py \
  --source-revision <dataset-commit-or-snapshot-id> \
  --motion-index 0 \
  --speed 1.0
```

Keep play terrain equal to train terrain for default variants.  For AMP, keep the
prepared demo directory available because the AMP algorithm checks the source
gate during runner construction.

## Sim2Real / Unitree simulator

Get-up is contact-rich.  Validate in Python play and Unitree simulator before any
real-robot attempt.

1. Copy the exported AMP policy into the existing GetUp deployment bundle:

   ```bash
   mkdir -p deploy/robots/g1_getup/config/policy/getup/v0/exported
   cp logs/rsl_rl/g1_getup_amp/<run>/policy.onnx \
     deploy/robots/g1_getup/config/policy/getup/v0/exported/policy.onnx
   ```

2. Build the controller:

   ```bash
   cmake -S deploy/robots/g1_getup -B deploy/robots/g1_getup/build
   cmake --build deploy/robots/g1_getup/build -j4
   ```

3. Start Unitree MuJoCo simulator and controller:

   ```bash
   ./simulate/build/unitree_mujoco
   ./deploy/robots/g1_getup/build/g1_getup_ctrl --network=lo --keyboard
   ```

4. Keyboard transitions are `f` then `g` (`Passive -> FixStand -> GetUp`).

Before hardware, confirm exported ONNX metadata, joint order, action scale,
default pose, PD gains, torque limits, and initial fallen geometry.  Keep an
operator ready to switch to Passive and physically support the robot on the first
motor-enable tests.

## Migration notes

- HoST reward groups are represented as MJLab reward/event wiring in
  `g1_getup/env_cfgs.py` plus reusable functions in `src/tasks/velocity/mdp/getup/`.
- Terrain variant metadata is stored on the env cfg (`getup_terrain`,
  `host_source_task`) so scripts and diagnostics can report provenance.
- AMP is not a renamed pose reward: it uses a discriminator, expert/policy
  transitions, and an AMP reward added into PPO only in the AMP task.
- Platform/wall/slope AMP are intentionally not enabled in this pass.
