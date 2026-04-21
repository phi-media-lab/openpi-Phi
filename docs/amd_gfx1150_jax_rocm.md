# AMD gfx1150 JAX ROCm Notes

These notes document the workaround needed to run JAX on the AMD Radeon 890M (`gfx1150`) iGPU in Docker on this machine.

## Machine Summary

- Host: Lenovo ThinkPad P16s Gen 4 AMD
- GPU: AMD Radeon 890M
- ISA: `gfx1150`
- OS: Ubuntu 24.04.3

## Observed Behavior

Inside `rocm/jax-community:rocm6.3.2-jax0.5.0-py3.12.8`, `rocminfo` sees the GPU correctly, but JAX fails by default with:

```text
RuntimeError: Unable to initialize backend 'rocm': INTERNAL: no supported devices found for platform ROCM
```

## Workaround

Set:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

This makes the `gfx1150` device behave as a `gfx1100`-compatible target for JAX ROCm plugin initialization.

## Repro Command

From the repo root:

```bash
scripts/docker/run_jax_rocm_gfx1150.sh   python -c "import jax; print('jax', jax.__version__); print('devices', jax.devices()); print('backend', jax.default_backend())"
```

## Validated Result

Observed result on this machine:

```text
jax 0.5.0
devices [RocmDevice(id=0)]
backend gpu
```

A minimal GPU matmul also succeeded:

```text
matmul_shape (1024, 1024)
matmul_sum 1073741824.0
```

## Limits

- This is a compatibility workaround, not a statement of official `gfx1150` support by prebuilt `rocm-jax` wheels.
- Host-side JAX on this machine still falls back to CPU.
- The validated path is Docker-based JAX ROCm with the override above.
