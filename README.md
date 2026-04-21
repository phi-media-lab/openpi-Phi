# openpi-Phi: MI300X Bring-Up Branch

This branch, `mi300x-bringup`, is a focused working branch for bringing `openpi` up on an AMD MI300X machine and documenting the validated JAX-on-ROCm training and inference path.

It is branch-specific by design. It does not try to be the generic upstream project homepage.

## What This Branch Changes

This branch currently adds or changes three things:

1. A training-path fix for fake/debug data that avoids an unnecessary hard dependency on `lerobot` during `debug` runs.
2. A reproducible policy inference benchmark helper.
3. MI300X / ROCm / JAX bring-up notes based on one validated machine.

## What Was Validated

### JAX on ROCm via Docker

The working JAX ROCm image used during validation was:

```bash
rocm/jax-community:rocm6.3.2-jax0.5.0-py3.12.8
```

Required container flags on the validation machine:

```bash
--device=/dev/kfd \
--device=/dev/dri \
--security-opt seccomp=unconfined \
--group-add "$(getent group render | cut -d: -f3)"
```

Observed minimal validation:

```text
jax 0.5.0
devices [RocmDevice(id=0)]
default_backend gpu
```

### Training Sanity Check

After making `lerobot` a lazy import for the fake/debug data path, the following command completed one training step successfully:

```bash
python scripts/train.py debug --num-train-steps 1
```

Observed result:

```text
Step 0: grad_norm=10.1185, loss=2.6070, param_norm=477.8696
```

### pi05 Inference

Validated paths:

- single `policy.infer()` for `pi05_aloha`
- `ActionChunkBroker`
- `serve_policy.py` with a full local `openpi`-style checkpoint directory

### Steady-State Inference Throughput

Validated steady-state benchmark result for `pi05_aloha` on the validation MI300X machine:

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

## Validated Environment

Validated machine characteristics:

- GPU: `AMD Instinct MI300X VF`
- GPU arch: `gfx942`
- OS: Ubuntu 24.04.3
- Host ROCm driver observed during bring-up: `6.14.14`

## Key Files on This Branch

- [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py)
- [scripts/benchmark_policy.py](scripts/benchmark_policy.py)
- [docs/mi300x_bringup.md](docs/mi300x_bringup.md)

## Full Bring-Up Notes

For the step-by-step environment setup, checkpoint handling, validation commands, and observed logs, see:

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

Validated checkpoint layout:

```text
<OPENPI_DATA_HOME>/openpi-assets/checkpoints/pi05_base
```

On the validation machine, this resolved to:

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

- This README is intentionally branch-specific.
- For generic project usage, installation, checkpoints, and examples, refer to the upstream `openpi` README.
