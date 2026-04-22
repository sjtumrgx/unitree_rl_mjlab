# Unitree G1 Topology Get-Up Task Notes

This document tracks the new **terrain-indexed get-up** task family that is being built
**without changing the existing `Unitree-G1-AntiFall-*` training tasks**.

## New task IDs

- `Unitree-G1-TopologyGetUp-Stage0`
- `Unitree-G1-TopologyGetUp-Benchmark`
- `Unitree-G1-TopologyGetUp-Stage0-NaiveDepth`
- `Unitree-G1-TopologyGetUp-Stage0-Teacher`
- `Unitree-G1-TopologyGetUp-Stage0-Distill`

These IDs are registered independently from the anti-fall ladder.

## Current contract

### Main claim surface
- get-up only
- terrain/support-topology transfer
- mandatory onboard depth
- onboard-only deployment inputs

### Current student observation split
- `actor`: existing proprioceptive contract retained for compatibility
- `camera`: mandatory `support_depth` observation group

### Current deploy/export contract
- SGI v1 metadata is emitted through the topology-getup runner helpers
- a dedicated `deploy.yaml` template is generated under:
  - `deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml`
- deploy-side observation registry now includes `support_depth`
- topology-getup deploy/runtime ownership now lives under `deploy/robots/g1_getup`
- default `deploy/robots/g1` remains velocity-oriented and no longer hosts topology-getup policy/config ownership
- the dedicated get-up deploy runtime now contains an isolated `SupportGeometryProvider` with
  organized-PointCloud2 raster projection, configurable timeout, and last-valid-frame retention

## Important boundary

The current implementation is intentionally additive:
- it does **not** rename, replace, or mutate the existing anti-fall task IDs
- anti-fall registration/tests remain part of the regression suite

## Remaining gaps

- the real robot still needs the final topic hookup / calibration path validated end-to-end;
  the deploy provider now handles organized PointCloud2 ingestion and dropout retention, but
  the physical robot feed still needs the final live integration pass
- the explicit teacher-student + topology bottleneck training loop is only partially
  scaffolded; the runner/config path is in place, but the end-to-end research method is
  still being completed
- get-up-specific benchmark/evaluation is currently a first-pass harness and still needs
  richer held-out topology trial logic

## Distillation lane

- `Unitree-G1-TopologyGetUp-Stage0-Distill` is a dedicated teacher-student training lane.
- The student uses the same mandatory depth-conditioned SGI contract as the deployable model.
- The teacher consumes the richer `critic + camera` observation set.
- Training requires a teacher checkpoint path to be provided; the distillation runner will refuse to train without one.

## Distillation wrapper

Use the dedicated wrapper to start the teacher-student lane without touching existing tasks:

```bash
python scripts/train_topology_getup_distill.py \
  --teacher-checkpoint path/to/teacher.pt -- \
  --agent.max-iterations=5000 --env.scene.num-envs=4096
```

This forwards into `scripts/train.py Unitree-G1-TopologyGetUp-Stage0-Distill` and injects `--agent.teacher-load-path=...`.

The wrapper also now accepts a **teacher run directory** instead of a raw checkpoint path:

```bash
python scripts/train_topology_getup_distill.py \
  --teacher-run-dir path/to/teacher_run -- \
  --agent.max-iterations=5000 --env.scene.num-envs=4096
```

When given a run directory, it first looks for `topology_getup_artifacts.json` and falls back to the latest
`model_*.pt` checkpoint if needed.

## Standardized artifact manifest

Topology-getup export lanes now emit `topology_getup_artifacts.json` next to the exported policy artifacts.
This manifest standardizes the teacher → distill handoff and records:

- lane (`teacher`, `main`, `naive_depth`, `distill`)
- checkpoint
- `policy.onnx`
- `policy_analysis.onnx`
- `params/deploy.yaml`
- SGI version
- distillation metadata / teacher checkpoint source when applicable

## Deploy-ready artifact promotion

Use the promotion helper to stage a deployable student bundle into the dedicated `g1_getup` runtime tree:

```bash
python scripts/promote_topology_getup_artifact.py \
  --run-dir path/to/topology_getup_run
```

This copies the selected run-local artifacts into:

- `deploy/robots/g1_getup/config/policy/topology_getup/v0/exported/policy.onnx`
- `deploy/robots/g1_getup/config/policy/topology_getup/v0/exported/policy_analysis.onnx`
- `deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml`

The helper rejects the teacher lane as non-deployable; use `main`, `naive_depth`, or `distill` run artifacts.

## Seen-vs-heldout terrain split

- Training stage uses a **seen topology mix**: `flat`, `pyramid_stairs`, `hf_pyramid_slope`, `random_rough`.
- Benchmark task switches to a **held-out mix**: `open_stairs`, `random_stairs`, `random_spread_boxes`.
- Held-out benchmark summaries now keep **family-specific buckets** instead of collapsing everything into one generic held-out bucket:
  - `stair-height-heldout`
  - `edge-geometry-heldout`
  - `support-arrangement-heldout`
- The get-up task also disables the inherited `command_vel` curriculum so the zero-command get-up objective is not overwritten by locomotion curriculum updates.

## Support-contact / clearance signals

The topology-getup lane now adds a dedicated `support_body_contact` terrain-contact sensor for non-foot support bodies.
This is used to surface:
- `support_contact_pattern` in critic observations
- `support_body_contact_count` in metrics
- `torso_clearance` in metrics

These signals are additive and do not modify the existing anti-fall task contracts.

## Get-up-specific rewards and metrics

The topology-getup lane now adds dedicated get-up shaping and evaluation surfaces on the new task family only:

### Rewards
- `getup_posture_reward`
- `support_contact_diversity_reward`
- `pelvis_clearance_penalty`
- `getup_completion_bonus`

### Metrics
- `support_body_contact_count`
- `torso_clearance`
- `getup_upright`
- `getup_success_count`
- `getup_latency`
- `pelvis_clearance_violation`

These are additive-only and do not change the anti-fall reward/metric contracts.

## Robot validation checklist CLI

Use the benchmark wrapper to print the current real-robot trial matrix:

```bash
python scripts/benchmark_topology_getup.py robot-checklist
```

This emits the staged checklist used by the topology-getup lane for preflight, tethered trials, and degraded-depth stress checks.

## Depth topic configuration helper

When the real robot depth topic is known, patch the deploy contract explicitly:

```bash
python scripts/configure_topology_getup_depth_topic.py \
  --topic-name /your/depth/points \
  --pointcloud-mode euclidean_norm \
  --timeout-ms 500 \
  --retain-last-valid-frame
```

This only updates the dedicated topology-getup `deploy.yaml`; it does not modify the existing anti-fall deploy paths.

## Depth capture inspection helper

When you have recorded `support_depth` captures (for example in NPZ form), inspect them and save visual calibration artifacts with:

```bash
python scripts/inspect_topology_getup_depth_capture.py \
  path/to/capture.npz \
  --deploy-yaml deploy/robots/g1_getup/config/policy/topology_getup/v0/params/deploy.yaml \
  --output depth_capture_summary.json \
  --artifact-dir depth_capture_artifacts
```

This emits summary statistics and saves:
- `first_frame.png`
- `last_frame.png`
- `mean_frame.png`

## Head-contact guard

The topology-getup lane now removes the inherited `fell_over` termination and replaces it with a dedicated `head_contact` guard.
This matters because the new task intentionally resets the robot into fallen poses, so the old upright-only termination would end episodes immediately.
The head-contact guard keeps the get-up task trainable while still terminating obviously unsafe head-ground impacts.

## Termination policy

The new topology-getup lane now removes the inherited `is_terminated` reward and the generic `fell_over` termination from the underlying locomotion scaffold.
For get-up this matters because the robot starts from fallen poses by design. The lane instead uses a dedicated `head_contact` guard so unsafe head-ground impacts still terminate, while normal fallen-pose starts remain trainable.

## Naive depth baseline lane

- `Unitree-G1-TopologyGetUp-Stage0-NaiveDepth` is the explicit deployable depth-conditioned baseline.
- It uses the same topology-getup env family and mandatory depth observation contract.
- Unlike the main method, it uses a plain `SpatialSoftmaxCNNModel` without the topology bottleneck latent.

## Teacher lane

- `Unitree-G1-TopologyGetUp-Stage0-Teacher` is the richer teacher PPO lane.
- It uses `critic + camera` observations for both actor and critic.
- Use the wrapper below to train it without touching existing tasks:

```bash
python scripts/train_topology_getup_teacher.py -- --agent.max-iterations=5000 --env.scene.num-envs=4096
```

## Baseline margin checker

Use the benchmark wrapper to compare the main method against the naive depth baseline after evaluation summaries are produced:

```bash
python scripts/benchmark_topology_getup.py compare-summary \
  path/to/main_summary.json path/to/naive_summary.json
```

This reports aggregate and per-heldout-bucket success-rate deltas using the thresholds from the planning spec.
The comparison now requires the aggregate margin **and every held-out family bucket** to pass, not just one bucket.
If a required held-out family bucket is missing from the evaluation summary, `compare-summary` now fails the comparison.

## Experiment suite planner

To print the canonical topology-getup lane inventory:

```bash
python scripts/benchmark_topology_getup.py suite-plan \
  --teacher-checkpoint path/to/teacher.pt \
  --iterations 5000 --num-envs 4096
```

This emits the additive experiment lanes without touching the existing anti-fall tasks.
Read the output as a dependency-aware lane set, not a strict serial order:

- `teacher -> distill`
- `main`
- `naive_depth`

`distill` depends on the teacher artifact, while `main` and `naive_depth` are
independent deployable baselines.

## Scenario summary aggregator

If you have per-scenario benchmark results, aggregate them into the held-out summary schema expected by `compare-summary`:

```bash
python scripts/benchmark_topology_getup.py aggregate-summary path/to/results.json
```

This computes weighted aggregate success and per-bucket held-out summaries.

## Topology latent analysis export

The main and distillation runners now also export `policy_analysis.onnx`, which emits:
- `actions`
- `topology_latent`

This is intended for mechanism analysis (for example, latent clustering or topology-family visualization) without changing the deployable policy contract.

## Latent summary helper

Use the analysis helper on exported topology-latent datasets to summarize cluster separation and nearest-centroid consistency:

```bash
python scripts/analyze_topology_latent.py \
  path/to/latents.npz \
  --output latent_summary.json \
  --plot-dir latent_plots
```

Expected NPZ fields:
- `topology_latent`: `[N, D]` float array
- `bucket`: `[N]` bucket / topology family labels

When `--plot-dir` is supplied, the helper also saves:
- `centroid_distance_heatmap.png`
- `within_scatter.png`

## Smoke-train commands

Minimal CPU smoke checks used during this Ralph session:

```bash
python scripts/train_topology_getup_main.py -- --agent.max-iterations=1 --env.scene.num-envs=1 --gpu-ids cpu --agent.logger tensorboard --agent.upload-model False
python scripts/train_topology_getup_naive.py -- --agent.max-iterations=1 --env.scene.num-envs=1 --gpu-ids cpu --agent.logger tensorboard --agent.upload-model False
python scripts/train_topology_getup_teacher.py -- --agent.max-iterations=1 --env.scene.num-envs=1 --gpu-ids cpu --agent.logger tensorboard --agent.upload-model False
python scripts/train_topology_getup_distill.py --teacher-checkpoint path/to/teacher.pt -- --agent.max-iterations=1 --env.scene.num-envs=1 --gpu-ids cpu --agent.logger tensorboard --agent.upload-model False
```

In this session, main / naive / teacher / distill all completed a 1-iteration CPU smoke run successfully.
