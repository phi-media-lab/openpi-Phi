# MI300X Bring-Up Notes

This document captures the validated bring-up flow for running `openpi` on an AMD MI300X machine with ROCm JAX in Docker, plus the latest training and inference checks that were run during setup.

## Machine Summary

- GPU: `AMD Instinct MI300X VF`
- GPU arch: `gfx942`
- OS: Ubuntu 24.04.3
- ROCm driver observed on host: `6.14.14`

## Repository Layout Used During Bring-Up

- Upstream working tree: `/mnt/models_alehe/phi-fbsh/openpi`
- Fork working tree: `/mnt/models_alehe/phi-fbsh/openpi-Phi`
- Local cached checkpoint root: `/mnt/models_alehe/phi-fbsh/openpi-cache`

## What Worked and What Did Not

### PyTorch on ROCm

PyTorch on ROCm worked on the host. `torch.cuda.is_available()` was true and the MI300X device was visible.

### JAX on ROCm on the host

Installing ROCm JAX wheels directly on the host did not produce a working runtime. The main failure was a HIP runtime mismatch (`libamdhip64.so.6` expected by the wheel was not available on the host in that form).

### JAX on ROCm in Docker

JAX on ROCm worked in Docker with the official AMD community image:

```bash
rocm/jax-community:rocm6.3.2-jax0.5.0-py3.12.8
```

Required container flags:

```bash
--device=/dev/kfd \
--device=/dev/dri \
--security-opt seccomp=unconfined \
--group-add 992
```

The `render` group on the machine had GID `992` during validation.

## Minimal JAX ROCm Validation

A minimal JAX validation in the container showed:

```text
jax 0.5.0
devices [RocmDevice(id=0)]
default_backend gpu
```

## openpi Training Validation

The first training blocker for `scripts/train.py debug --num-train-steps 1` was an eager top-level `lerobot` import in `src/openpi/training/data_loader.py`.

`debug` uses fake data and should not need `lerobot`, so the file was updated to import `lerobot` only when a real LeRobot dataset is requested.

After that fix, the following command completed one training step successfully in the ROCm JAX container:

```bash
python scripts/train.py debug --num-train-steps 1
```

Observed result:

```text
Step 0: grad_norm=10.1185, loss=2.6070, param_norm=477.8696
```

A checkpoint was written to:

```text
checkpoints/debug/debug/0
```

## openpi pi05 Inference Validation

### Single inference

`pi05_aloha` inference was validated against the base pi05 checkpoint.

- Config: `pi05_aloha`
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_base`

The single-call inference path returned successfully with action shape `(50, 14)`.

### ActionChunkBroker

`ActionChunkBroker` was also validated successfully for `pi05_aloha`.

Observed result:

```text
broker_calls 50
broker_unique_shapes [(14,)]
```

## Getting a Serveable openpi-Style pi05 Checkpoint Locally

The Hugging Face `lerobot/pi05_base` snapshot is not enough for the JAX serving path because `serve_policy.py` expects an `openpi`-style checkpoint directory with `params/` and `assets/`.

A complete local checkpoint was downloaded with anonymous GCS access using `openpi.shared.download`:

```python
from openpi.shared import download

path = download.maybe_download(
    "gs://openpi-assets/checkpoints/pi05_base",
    gs={"token": "anon"},
)
print(path)
```

Validated local checkpoint path:

```text
/mnt/models_alehe/phi-fbsh/openpi-cache/openpi-assets/checkpoints/pi05_base
```

Validated directory contents included:

```text
assets/
params/_METADATA
params/manifest.ocdbt
params/_CHECKPOINT_METADATA
```

## serve_policy Validation

The following command successfully brought up the policy server with the local checkpoint:

```bash
PYTHONPATH=/workspace/openpi/src /pyenv/versions/3.12.8/bin/python \
  scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config pi05_aloha \
  --policy.dir /workspace/pi05_base
```

Observed logs included:

```text
Finished restoring checkpoint in 4.79 seconds from /workspace/pi05_base/params.
Loaded norm stats from /workspace/pi05_base/assets/trossen
Creating server (host: ..., ip: ...)
server listening on 0.0.0.0:8010
```

## Steady-State pi05 Inference Throughput

A steady-state benchmark was run with:

- one container startup
- one model load
- 10 warmup calls
- 100 measured `policy.infer()` calls
- config `pi05_aloha`
- checkpoint `/workspace/pi05_base`

Results:

```text
avg_sec 0.10943090332672
median_sec 0.10942615056410432
p90_sec 0.1103091649711132
p95_sec 0.11072379071265459
min_sec 0.1073203468695283
max_sec 0.11529319919645786
hz_avg 9.138186468354116
hz_median 9.138583371935187
hz_p90 9.065429878486263
chunk_hz_avg 456.90932341770576
chunk_hz_median 456.9291685967594
```

Interpretation:

- Full policy re-inference runs at about `9.1 Hz`.
- Since `action_horizon = 50`, the action chunk emission rate is about `457 steps/s`.

## Reproducible Benchmark Command

After downloading the full checkpoint locally, a reproducible benchmark can be run with:

```bash
PYTHONPATH=/workspace/openpi/src /pyenv/versions/3.12.8/bin/python \
  scripts/benchmark_policy.py \
  --config pi05_aloha \
  --checkpoint-dir /workspace/pi05_base \
  --example aloha \
  --warmup 10 \
  --runs 100
```

## Recommended Next Commits on This Branch

1. Keep the `data_loader.py` lazy `lerobot` import fix as a standalone commit.
2. Keep this bring-up note as a documentation commit.
3. Add a reproducible benchmark/helper script in a follow-up commit.
