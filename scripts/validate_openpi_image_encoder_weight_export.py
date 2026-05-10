import dataclasses
import json
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    golden: str = "build/openpi_golden/pi05_aloha_base_0_rgb_image_encoder.npz"
    weights: str = "build/openpi_golden/pi05_base_image_encoder_weights_bfloat16_first_1_layers.npz"
    strict: bool = False
    stem_max_abs_tol: float = 1e-4
    posemb_max_abs_tol: float = 1e-4
    head_max_abs_tol: float = 1.25e-1


def load_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parse_json_scalar(array: np.ndarray) -> dict[str, Any]:
    return json.loads(str(array.item()))


def patch_embed(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, height, width, channels = image.shape
    patch_h, patch_w, kernel_channels, _ = kernel.shape
    if channels != kernel_channels:
        raise ValueError(f"Image channels {channels} do not match kernel channels {kernel_channels}")
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(f"Image shape {(height, width)} is not divisible by patch shape {(patch_h, patch_w)}")

    grid_h = height // patch_h
    grid_w = width // patch_w
    patches = image.reshape(batch, grid_h, patch_h, grid_w, patch_w, channels)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    return np.einsum("bhwpqc,pqco->bhwo", patches, kernel, optimize=True) + bias


def head_project_bfloat16(encoded: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    encoded_jax = jnp.asarray(encoded, dtype=jnp.bfloat16)
    kernel_jax = jnp.asarray(kernel, dtype=jnp.bfloat16)
    bias_jax = jnp.asarray(bias, dtype=jnp.bfloat16)
    projected = jnp.einsum("btd,do->bto", encoded_jax, kernel_jax, optimize=True) + bias_jax
    return np.asarray(jax.device_get(projected), dtype=np.float32)


def summarize(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float32) - expected.astype(np.float32)
    metrics = {
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff * diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
    }
    print(
        f"{name} max_abs={metrics['max_abs']:.8g} "
        f"rms={metrics['rms']:.8g} mean_abs={metrics['mean_abs']:.8g}"
    )
    return metrics


def maybe_fail(strict: bool, name: str, metrics: dict[str, float], tolerance: float) -> None:
    if strict and metrics["max_abs"] > tolerance:
        raise SystemExit(f"{name} max_abs {metrics['max_abs']} exceeds tolerance {tolerance}")


def main(args: Args) -> None:
    golden_path = pathlib.Path(args.golden)
    weights_path = pathlib.Path(args.weights)
    golden = load_npz(str(golden_path))
    weights = load_npz(str(weights_path))

    golden_metadata = parse_json_scalar(golden["metadata_json"])
    weight_manifest = parse_json_scalar(weights["manifest_json"])
    print(f"golden {golden_path}")
    print(f"weights {weights_path}")
    print(f"golden_tokens_shape {tuple(golden['tokens'].shape)}")
    print(f"weight_restore_dtype {weight_manifest.get('restore_dtype')}")
    print(f"weight_layer_limit {weight_manifest.get('layer_limit')}")
    print(f"golden_camera {golden_metadata.get('camera')}")

    stem = patch_embed(golden["image"], weights["patch_embed_kernel"], weights["patch_embed_bias"])
    stem_metrics = summarize("patch_embed_vs_intermediate_stem", stem, golden["intermediate_stem"])
    maybe_fail(args.strict, "patch_embed_vs_intermediate_stem", stem_metrics, args.stem_max_abs_tol)

    with_posemb = stem.reshape(stem.shape[0], stem.shape[1] * stem.shape[2], stem.shape[3]) + weights["pos_embedding"]
    posemb_metrics = summarize("posemb_vs_intermediate_with_posemb", with_posemb, golden["intermediate_with_posemb"])
    maybe_fail(args.strict, "posemb_vs_intermediate_with_posemb", posemb_metrics, args.posemb_max_abs_tol)

    head_tokens = head_project_bfloat16(golden["intermediate_encoded"], weights["head_kernel"], weights["head_bias"])
    head_metrics = summarize("head_bfloat16_vs_tokens", head_tokens, golden["tokens"])
    maybe_fail(args.strict, "head_bfloat16_vs_tokens", head_metrics, args.head_max_abs_tol)


if __name__ == "__main__":
    main(tyro.cli(Args))
