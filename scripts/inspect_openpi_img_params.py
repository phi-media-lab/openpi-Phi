import dataclasses
import pathlib
from typing import Any

import orbax.checkpoint as ocp
import tyro

from openpi.shared import download


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base"
    prefix: str = "PaliGemma/img"
    max_rows: int = 300


def flatten(tree: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if isinstance(tree, dict):
        rows: list[tuple[tuple[str, ...], Any]] = []
        for key, value in tree.items():
            rows.extend(flatten(value, (*prefix, str(key))))
        return rows
    return [(prefix, tree)]


def strip_value_suffix(path: tuple[str, ...]) -> tuple[str, ...]:
    if path and path[-1] == "value":
        return path[:-1]
    return path


def shape_of(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "-"
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return "-" if dtype is None else str(dtype)


def classify(path: str, shape: str) -> str:
    if path.startswith("PaliGemma/"):
        path = path.removeprefix("PaliGemma/")
    if path == "img/embedding/kernel":
        return "vit patch embedding kernel"
    if path == "img/embedding/bias":
        return "vit patch embedding bias"
    if path == "img/pos_embedding":
        return "vit positional embedding"
    if "LayerNorm_0" in path:
        return "vit pre-attention layernorm"
    if "LayerNorm_1" in path:
        return "vit pre-mlp layernorm"
    if "MultiHeadDotProductAttention_0/query" in path:
        return "vit query projection"
    if "MultiHeadDotProductAttention_0/key" in path:
        return "vit key projection"
    if "MultiHeadDotProductAttention_0/value" in path:
        return "vit value projection"
    if "MultiHeadDotProductAttention_0/out" in path:
        return "vit attention output projection"
    if "MlpBlock_0/Dense_0" in path:
        return "vit mlp up projection"
    if "MlpBlock_0/Dense_1" in path:
        return "vit mlp down projection"
    if path == "img/Transformer/encoder_norm/scale":
        return "vit final norm scale"
    if path == "img/Transformer/encoder_norm/bias":
        return "vit final norm bias"
    if path == "img/head/kernel":
        return "image-token projection kernel"
    if path == "img/head/bias":
        return "image-token projection bias"
    return "unclassified"


def main(args: Args) -> None:
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    params_dir = checkpoint_dir / "params"
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_dir)

    if hasattr(metadata, "tree"):
        metadata = metadata.tree
    params = metadata["params"] if isinstance(metadata, dict) and "params" in metadata else metadata
    rows = []
    for path_tuple, value in flatten(params):
        path_tuple = strip_value_suffix(path_tuple)
        path = "/".join(path_tuple)
        if not path.startswith(args.prefix):
            continue
        shape = shape_of(value)
        rows.append((path, shape, dtype_of(value), classify(path, shape)))

    print(f"checkpoint_dir {checkpoint_dir}")
    print(f"params_dir {params_dir}")
    print(f"prefix {args.prefix}")
    print(f"matched_rows {len(rows)}")
    print()
    print("| path | shape | dtype | note |")
    print("| --- | --- | --- | --- |")
    for path, shape, dtype, note in rows[: args.max_rows]:
        print(f"| `{path}` | `{shape}` | `{dtype}` | {note} |")
    if len(rows) > args.max_rows:
        print(f"\ntruncated_rows {len(rows) - args.max_rows}")


if __name__ == "__main__":
    main(tyro.cli(Args))
