# openpi-Phi: MI300X Bring-Up Branch

This branch, `mi300x-bringup`, is a focused working branch for bringing `openpi` up on an AMD MI300X machine and documenting the validated JAX-on-ROCm inference/training path.

It intentionally does **not** try to preserve the full generic upstream README. Instead, it only describes what was actually validated and added on this branch.

## Scope of This Branch

This branch currently contains three things:

1. A training-path fix for fake/debug data that avoids an unnecessary hard dependency on `lerobot` during `debug` runs.
2. A branch-specific MI300X/ROCm/JAX bring-up document.
3. A reproducible policy inference benchmark helper.

## Branch Commits

Current branch commits:

- `2d86a2f` `fix: lazily import lerobot for fake/debug data path`
- `14f360e` `tools: add reproducible policy benchmark helper`

## Validated Target Environment

Validated machine characteristics:

- GPU: `AMD Instinct MI300X VF`
- GPU arch: `gfx942`
- OS: Ubuntu 24.04.3
- Host ROCm driver observed during bring-up: `6.14.14`

## What Was Validated

### 1. JAX on ROCm via Docker

The working JAX ROCm image used during validation was:

```bash
rocm/jax-community:rocm6.3.2-jax0.5.0-py3.12.8
```

Required Docker flags:

```bash
--device=/dev/kfd \
--device=/dev/dri \
--security-opt seccomp=unconfined \
--group-add 992
```

Observed minimal validation:

```text
jax 0.5.0
devices [RocmDevice(id=0)]
default_backend gpu
```

### 2. Training sanity check

After making `lerobot` a lazy import for the fake/debug data path, the following command successfully completed one training step:

```bash
python scripts/train.py debug --num-train-steps 1
```

Observed result:

```text
Step 0: grad_norm=10.1185, loss=2.6070, param_norm=477.8696
```

### 3. pi05 inference

Validated paths:

- single `policy.infer()` for `pi05_aloha`
- `ActionChunkBroker`
- `serve_policy.py` with a local full `openpi`-style checkpoint directory

### 4. Steady-state inference throughput

Validated steady-state benchmark result for `pi05_aloha` on MI300X:

```text
avg_sec 0.10943090332672
median_sec 0.10942615056410432
p90_sec 0.1103091649711132
p95_sec 0.11072379071265459
hz_avg 9.138186468354116
hz_median 9.138583371935187
chunk_hz_avg 456.90932341770576
```

Interpretation:

- full policy re-inference is about `9.1 Hz`
- since `action_horizon = 50`, the action chunk emission rate is about `457 steps/s`

## Key Files Added or Changed on This Branch

- [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py)
- [docs/mi300x_bringup.md](docs/mi300x_bringup.md)
- [scripts/benchmark_policy.py](scripts/benchmark_policy.py)

## Full Bring-Up Notes

For the detailed step-by-step environment history, checkpoint handling, validation commands, and observed logs, see:

- [docs/mi300x_bringup.md](docs/mi300x_bringup.md)

## Reproducible Benchmark

After downloading a full `openpi`-style checkpoint locally, run:

```bash
PYTHONPATH=/workspace/openpi/src /pyenv/versions/3.12.8/bin/python \
  scripts/benchmark_policy.py \
  --config pi05_aloha \
  --checkpoint-dir /workspace/pi05_base \
  --example aloha \
  --warmup 10 \
  --runs 100
```

## Local Checkpoint Layout That Worked

The JAX serving path requires a full `openpi` checkpoint directory, not just a Hugging Face snapshot with `model.safetensors`.

Validated local checkpoint root:

```text
/mnt/models_alehe/phi-fbsh/openpi-cache/openpi-assets/checkpoints/pi05_base
```

Expected structure includes:

```text
assets/
params/_METADATA
params/manifest.ocdbt
params/_CHECKPOINT_METADATA
```

## serve_policy Example

Validated command shape:

```bash
PYTHONPATH=/workspace/openpi/src /pyenv/versions/3.12.8/bin/python \
  scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config pi05_aloha \
  --policy.dir /workspace/pi05_base
```

## Notes

- This branch README is intentionally branch-specific.
- If this branch later grows beyond MI300X bring-up and validation work, the README should be revisited.
- Upstream-generic usage information should stay in upstream docs or be reintroduced only when it is directly relevant to this branch.
