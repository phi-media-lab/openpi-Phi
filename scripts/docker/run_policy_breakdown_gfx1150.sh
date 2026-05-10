#!/usr/bin/env bash
# Runs the policy breakdown benchmark in the local ROCm/JAX gfx1150 container.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-openpi-jax-rocm-gfx1150:latest}"
OPENPI_DATA_HOME_HOST="${OPENPI_DATA_HOME_HOST:-${HOME}/.cache/openpi}"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=8g \
  -e PYENV_VERSION=3.12.8 \
  -e OPENPI_DATA_HOME=/home/amd/.cache/openpi \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=src:packages/openpi-client/src \
  -v "${ROOT_DIR}:/workspace/openpi" \
  -v "${OPENPI_DATA_HOME_HOST}:/home/amd/.cache/openpi" \
  -w /workspace/openpi \
  "${IMAGE}" \
  python scripts/benchmark_policy_breakdown.py "$@"
