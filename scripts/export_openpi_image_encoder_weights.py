import dataclasses
import json
import pathlib
from typing import Any

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tyro

from openpi.shared import download


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    alias: str
    path: tuple[str, ...]
    layout: str
    layer_stacked: bool = False


PARAM_SPECS = (
    ParamSpec("patch_embed_kernel", ("embedding", "kernel"), "HWIO: patch_h, patch_w, rgb, width"),
    ParamSpec("patch_embed_bias", ("embedding", "bias"), "O: width"),
    ParamSpec("pos_embedding", ("pos_embedding",), "1, tokens, width"),
    ParamSpec(
        "block_ln0_scale",
        ("Transformer", "encoderblock", "LayerNorm_0", "scale"),
        "layer, width",
        True,
    ),
    ParamSpec(
        "block_ln0_bias",
        ("Transformer", "encoderblock", "LayerNorm_0", "bias"),
        "layer, width",
        True,
    ),
    ParamSpec(
        "block_q_kernel",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "query", "kernel"),
        "layer, width, heads, head_dim",
        True,
    ),
    ParamSpec(
        "block_q_bias",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "query", "bias"),
        "layer, heads, head_dim",
        True,
    ),
    ParamSpec(
        "block_k_kernel",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "key", "kernel"),
        "layer, width, heads, head_dim",
        True,
    ),
    ParamSpec(
        "block_k_bias",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "key", "bias"),
        "layer, heads, head_dim",
        True,
    ),
    ParamSpec(
        "block_v_kernel",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "value", "kernel"),
        "layer, width, heads, head_dim",
        True,
    ),
    ParamSpec(
        "block_v_bias",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "value", "bias"),
        "layer, heads, head_dim",
        True,
    ),
    ParamSpec(
        "block_attn_out_kernel",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "out", "kernel"),
        "layer, heads, head_dim, width",
        True,
    ),
    ParamSpec(
        "block_attn_out_bias",
        ("Transformer", "encoderblock", "MultiHeadDotProductAttention_0", "out", "bias"),
        "layer, width",
        True,
    ),
    ParamSpec(
        "block_ln1_scale",
        ("Transformer", "encoderblock", "LayerNorm_1", "scale"),
        "layer, width",
        True,
    ),
    ParamSpec(
        "block_ln1_bias",
        ("Transformer", "encoderblock", "LayerNorm_1", "bias"),
        "layer, width",
        True,
    ),
    ParamSpec(
        "block_mlp_up_kernel",
        ("Transformer", "encoderblock", "MlpBlock_0", "Dense_0", "kernel"),
        "layer, width, mlp_dim",
        True,
    ),
    ParamSpec(
        "block_mlp_up_bias",
        ("Transformer", "encoderblock", "MlpBlock_0", "Dense_0", "bias"),
        "layer, mlp_dim",
        True,
    ),
    ParamSpec(
        "block_mlp_down_kernel",
        ("Transformer", "encoderblock", "MlpBlock_0", "Dense_1", "kernel"),
        "layer, mlp_dim, width",
        True,
    ),
    ParamSpec(
        "block_mlp_down_bias",
        ("Transformer", "encoderblock", "MlpBlock_0", "Dense_1", "bias"),
        "layer, width",
        True,
    ),
    ParamSpec("encoder_norm_scale", ("Transformer", "encoder_norm", "scale"), "width"),
    ParamSpec("encoder_norm_bias", ("Transformer", "encoder_norm", "bias"), "width"),
    ParamSpec("head_kernel", ("head", "kernel"), "width, decoder_width"),
    ParamSpec("head_bias", ("head", "bias"), "decoder_width"),
)


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base"
    output: str = ""
    layer_limit: int = 0
    restore_dtype: str = "bfloat16"
    compressed: bool = True


def metadata_tree(metadata: Any) -> Any:
    return metadata.tree if hasattr(metadata, "tree") else metadata


def strip_value_suffix(tree: Any) -> Any:
    flat = traverse_util.flatten_dict(tree)
    if flat and all(path[-1] == "value" for path in flat):
        flat = {path[:-1]: value for path, value in flat.items()}
    return traverse_util.unflatten_dict(flat)


def jax_dtype(name: str) -> jnp.dtype:
    match name:
        case "bfloat16":
            return jnp.bfloat16
        case "float32":
            return jnp.float32
        case _:
            raise ValueError(f"Unsupported restore_dtype {name!r}; expected 'bfloat16' or 'float32'.")


def metadata_to_restore_item(tree: Any, dtype: jnp.dtype) -> Any:
    return jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, dtype), tree)


def image_leaf_transforms(img_metadata: Any) -> dict[str, Any]:
    transforms = {}
    for path in traverse_util.flatten_dict(img_metadata):
        relative_path = "/".join(path)
        transforms[f"img/{relative_path}"] = ocp.transform_utils.Transform(
            original_key=f"params/PaliGemma/img/{relative_path}"
        )
    return transforms


def restore_img_params(params_dir: pathlib.Path, dtype: jnp.dtype) -> dict[str, Any]:
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = metadata_tree(ckptr.metadata(params_dir))
        img_metadata = metadata["params"]["PaliGemma"]["img"]
        item = {"img": metadata_to_restore_item(img_metadata, dtype)}
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=dtype),
            item,
        )
        transforms = image_leaf_transforms(img_metadata)
        restored = ckptr.restore(
            params_dir,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=restore_args,
                transforms=transforms,
                transforms_default_to_original=False,
            ),
        )
    return strip_value_suffix(restored["img"])


def get_nested(tree: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = tree
    for key in path:
        value = value[key]
    return value


def default_output_path(args: Args) -> pathlib.Path:
    suffix = "all_layers" if args.layer_limit <= 0 else f"first_{args.layer_limit}_layers"
    return pathlib.Path("build/openpi_golden") / f"pi05_base_image_encoder_weights_{args.restore_dtype}_{suffix}.npz"


def maybe_slice_layers(array: np.ndarray, spec: ParamSpec, layer_limit: int) -> np.ndarray:
    if layer_limit <= 0 or not spec.layer_stacked:
        return array
    if array.shape[0] < layer_limit:
        raise ValueError(f"{spec.alias} has only {array.shape[0]} layers, cannot export layer_limit={layer_limit}")
    return array[:layer_limit]


def main(args: Args) -> None:
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    params_dir = checkpoint_dir / "params" if checkpoint_dir.name != "params" else checkpoint_dir
    output_path = pathlib.Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    restore_dtype = jax_dtype(args.restore_dtype)
    img_params = restore_img_params(params_dir, restore_dtype)
    arrays: dict[str, Any] = {}
    manifest_arrays = []

    for spec in PARAM_SPECS:
        array = np.asarray(get_nested(img_params, spec.path), dtype=np.float32)
        source_shape = list(array.shape)
        array = maybe_slice_layers(array, spec, args.layer_limit)
        arrays[spec.alias] = array
        manifest_arrays.append(
            {
                "alias": spec.alias,
                "source_path": "PaliGemma/img/" + "/".join(spec.path),
                "layout": spec.layout,
                "source_shape": source_shape,
                "export_shape": list(array.shape),
                "export_dtype": str(array.dtype),
                "layer_stacked": spec.layer_stacked,
            }
        )

    manifest = {
        "checkpoint_dir": str(checkpoint_dir),
        "params_dir": str(params_dir),
        "restore_dtype": args.restore_dtype,
        "export_dtype": "float32 container; values reflect restore_dtype conversion",
        "layer_limit": args.layer_limit,
        "layer_limit_meaning": "0 means all layers; positive values export the first N stacked transformer layers",
        "array_count": len(manifest_arrays),
        "arrays": manifest_arrays,
        "notes": [
            "Weights are exported from the OpenPI PaliGemma/img subtree only.",
            "The OpenPI image head projects ViT width 1152 to decoder width 2048.",
            "Patch embedding kernel uses Flax HWIO layout.",
            "Attention kernels use Flax layout width, heads, head_dim for Q/K/V and heads, head_dim, width for output.",
        ],
    }
    arrays["manifest_json"] = np.array(json.dumps(manifest, sort_keys=True))

    if args.compressed:
        np.savez_compressed(output_path, **arrays)
    else:
        np.savez(output_path, **arrays)

    print(f"wrote {output_path}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "arrays"}, indent=2, sort_keys=True))
    print("| alias | shape | dtype |")
    print("| --- | ---: | --- |")
    for item in manifest_arrays:
        print(f"| `{item['alias']}` | `{tuple(item['export_shape'])}` | `{item['export_dtype']}` |")


if __name__ == "__main__":
    main(tyro.cli(Args))
