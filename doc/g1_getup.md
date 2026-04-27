# G1 GetUp: HoST Migration and Train → Play Notes

G1 GetUp is an added module that ports HoST-style get-up terrain variants into
MJLab while keeping the normal Unitree RL MJLab train/play workflow.

## Scope

| HoST source concept | MJLab entrypoint | Terrain flag |
| --- | --- | --- |
| `g1_ground` | `Unitree-G1-GetUp` | `ground` |
| `g1_platform` | `Unitree-G1-GetUp` | `platform` |
| `g1_wall` | `Unitree-G1-GetUp` | `wall` |
| `g1_slope` | `Unitree-G1-GetUp` | `slope` |

Core mapping lives in:

- `src/tasks/velocity/config/g1_getup/env_cfgs.py`
- `src/tasks/velocity/config/g1_getup/rl_cfg.py`
- `src/tasks/velocity/mdp/getup/`

## Train

Convenience wrapper:

```bash
python scripts/train_getup.py --terrain platform -- --env.scene.num-envs=4096
```

Generic form:

```bash
python scripts/train.py Unitree-G1-GetUp --getup-terrain=platform
```

The terrain flag affects terrain mix, reset state, assist force, and run naming.
Use separate runs for each terrain unless you intentionally test cross-terrain
transfer.

## Play

```bash
python scripts/play_getup.py --terrain platform -- \
  --checkpoint_file logs/rsl_rl/g1_getup/<run>/model_*.pt
```

Keep play terrain equal to train terrain for baseline validation.  Cross-terrain
play is useful only after the baseline terrain succeeds.

## Migration notes

- HoST reward groups are represented as MJLab reward/event wiring in
  `g1_getup/env_cfgs.py` plus reusable functions in `src/tasks/velocity/mdp/getup/`.
- Terrain variant metadata is stored on the env cfg (`getup_terrain`,
  `host_source_task`) so scripts and diagnostics can report provenance.
- Pull/assist behavior is represented through MJLab event logic rather than
  copying HoST runner code directly.

## Sim2Real cautions

Get-up behavior is contact-rich and sensitive to initial pose, terrain height,
and motor limits.  Before attempting hardware:

1. Validate the same checkpoint in Python play.
2. Check that the robot starts from the expected fallen pose for the selected terrain.
3. Use conservative motor enable procedures and physical support.
4. Avoid testing near walls/platforms on hardware until ground get-up is reliable.
5. Keep an operator ready to switch to Passive.
