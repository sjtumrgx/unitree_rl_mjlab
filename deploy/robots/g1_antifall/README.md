# G1 AntiFall Deploy Target

This target mirrors the existing G1 deploy controller but is scoped to anti-fall
policies.

## Policy layout

Place the deployable anti-fall artifacts under:

```text
config/policy/antifall/stage4b/v0/
├── exported/
│   └── policy.onnx
└── params/
    └── deploy.yaml
```

The training-side anti-fall runner now exports `params/deploy.yaml` next to
`policy.onnx`, so you can copy both files from a stage run directory such as:

```text
logs/rsl_rl/g1_antifall_curriculum/<run>/stages/05_stage4b/
```

## Build

```bash
cmake -S deploy/robots/g1_antifall -B deploy/robots/g1_antifall/build
cmake --build deploy/robots/g1_antifall/build -j4
```

## Run

Simulation:

```bash
deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=lo
```

Real robot:

```bash
deploy/robots/g1_antifall/build/g1_antifall_ctrl --network=<ethernet-iface>
```
