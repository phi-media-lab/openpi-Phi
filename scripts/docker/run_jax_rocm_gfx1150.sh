#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${JAX_ROCM_IMAGE:-rocm/jax-community:rocm6.3.2-jax0.5.0-py3.12.8}
GFX_OVERRIDE=${HSA_OVERRIDE_GFX_VERSION:-11.0.0}
RENDER_GID=$(getent group render | cut -d: -f3)

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ $# -eq 0 ]]; then
  CMD='export PATH=/pyenv/versions/3.12.8/bin:$PATH; exec bash'
else
  printf -v USER_CMD '%q ' "$@"
  CMD="export PATH=/pyenv/versions/3.12.8/bin:\$PATH; ${USER_CMD}"
fi

exec docker run --rm "${TTY_ARGS[@]}"   -e HSA_OVERRIDE_GFX_VERSION="${GFX_OVERRIDE}"   --device=/dev/kfd   --device=/dev/dri   --security-opt seccomp=unconfined   --group-add "${RENDER_GID}"   -v "${ROOT_DIR}:/workspace/openpi"   -w /workspace/openpi   "${IMAGE}" bash -lc "${CMD}"
