# openpi-Phi: AMD gfx1150 Bring-Up Branch

This branch, `amd-gfx1150-bringup`, is a focused working branch for bringing `openpi` up on an AMD Radeon 890M (`gfx1150`) machine and documenting the validated JAX-on-ROCm workaround, local checkpoint setup, and measured inference results.

It is branch-specific by design. It does not try to be the generic upstream project homepage.

## What This Branch Changes

This branch currently carries four branch-specific pieces of work:

1. A `gfx1150`-specific JAX ROCm Docker launcher.
2. A `gfx1150`-specific JAX ROCm Dockerfile.
3. Branch-local bring-up notes for AMD Radeon 890M / `gfx1150`.
4. The earlier fake/debug data path fix that allows `debug` training without a hard `lerobot` dependency.

## Why This Branch Exists

The target machine for this branch is not MI300X. It is a laptop-class AMD APU, so the branch is explicitly about bring-up on that smaller ROCm target:

- GPU: `AMD Radeon Graphics`
- architecture: `gfx1150`
- device family: `Radeon 890M`

On this machine:

- host-side PyTorch ROCm works
- host-side JAX falls back to CPU
- Docker JAX ROCm does not recognize `gfx1150` by default

The validated workaround is to run JAX ROCm in Docker with:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

## What Was Validated

### JAX on ROCm via Docker

The working JAX ROCm image used during validation was:

```bash
rocm/jax-community:rocm6.3.2-jax0.5.0-py3.12.8
```

Required container flags on this machine:

```bash
--device=/dev/kfd \
--device=/dev/dri \
--security-opt seccomp=unconfined \
--group-add "$(getent group render | cut -d: -f3)"
```

Additional required workaround:

```bash
-e HSA_OVERRIDE_GFX_VERSION=11.0.0
```

Observed minimal validation:

```text
jax 0.5.0
devices [RocmDevice(id=0)]
backend gpu
```

A minimal GPU matmul also succeeded.

### Training Sanity Check

The following JAX training sanity check completed successfully on this machine:

```bash
python scripts/train.py debug --num-train-steps 1
```

Observed result:

```text
Step 0: grad_norm=10.1201, loss=2.6070, param_norm=477.8696
```

### Debug Checkpoint Inference

A locally trained `debug` checkpoint was restored and used for JAX policy inference on GPU.

Observed result:

```text
action_shape (50, 32)
action_dtype float32
```

### pi05 Inference

Validated paths on this machine:

- single `policy.infer()` for `pi05_aloha`
- local full `openpi`-style `pi05_base` checkpoint restore
- local cached tokenizer restore from `big_vision/paligemma_tokenizer.model`

Observed single-inference result:

```text
actions_shape (50, 14)
actions_dtype float64
policy_metadata {'reset_pose': [0, -1.5, 1.5, 0, 0, 0]}
```

### Steady-State pi05 Throughput

Validated steady-state benchmark result for `pi05_aloha` on this `gfx1150` machine:

```text
avg_sec 1.3594153436576015
median_sec 1.3638111074687913
p90_sec 1.388828556984663
p95_sec 1.3907981489319354
hz_avg 0.7356103523956339
hz_median 0.7332393720241668
chunk_hz_avg 36.7805176197817
chunk_hz_median 36.66196860120834
```

Interpretation:

- full policy re-inference is about `0.74 Hz`
- since `action_horizon = 50`, the action chunk emission rate is about `36.8 steps/s`

## Validated Environment

Validated machine characteristics:

- CPU: `AMD Ryzen AI 9 HX PRO 370`
- GPU: `AMD Radeon Graphics`
- GPU arch: `gfx1150`
- OS: Ubuntu 24.04.3

## Key Files on This Branch

- [scripts/docker/Dockerfile.jax_rocm_gfx1150](scripts/docker/Dockerfile.jax_rocm_gfx1150)
- [scripts/docker/run_jax_rocm_gfx1150.sh](scripts/docker/run_jax_rocm_gfx1150.sh)
- [docs/amd_gfx1150_jax_rocm.md](docs/amd_gfx1150_jax_rocm.md)
- [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py)
- [scripts/benchmark_policy.py](scripts/benchmark_policy.py)

## Full Bring-Up Notes

For the step-by-step environment setup, workaround rationale, validation commands, and observed results, see:

- [docs/amd_gfx1150_jax_rocm.md](docs/amd_gfx1150_jax_rocm.md)

## Reproducible JAX ROCm Launcher

From the repo root:

```bash
scripts/docker/run_jax_rocm_gfx1150.sh \
  python -c "import jax; print('jax', jax.__version__); print('devices', jax.devices()); print('backend', jax.default_backend())"
```

## Local Checkpoint Layout That Worked

The JAX policy path requires a full `openpi` checkpoint directory, not just a Hugging Face snapshot with `model.safetensors`.

Validated checkpoint layout:

```text
/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base
```

Expected structure includes:

```text
assets/
params/_METADATA
params/manifest.ocdbt
params/_CHECKPOINT_METADATA
params/ocdbt.process_0/manifest.ocdbt
```

Tokenizer path that also had to exist locally on this machine:

```text
/home/amd/.cache/openpi/big_vision/paligemma_tokenizer.model
```

## Reproducible pi05 Benchmark

With the local checkpoint mounted into the validated Docker image, the benchmark command shape is:

```bash
PYTHONPATH=/workspace/openpi/src /pyenv/versions/3.12.8/bin/python \
  scripts/benchmark_policy.py \
  --config pi05_aloha \
  --checkpoint-dir /workspace/pi05_base \
  --example ALOHA \
  --warmup 10 \
  --runs 100
```

## Notes

- This README is intentionally branch-specific.
- The `gfx1150` JAX ROCm path here depends on `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- This branch documents a successful compatibility bring-up, not a claim of official `gfx1150` support by prebuilt ROCm JAX wheels.
- For generic project usage, installation, checkpoints, and examples, refer to the upstream `openpi` README.
