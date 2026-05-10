import dataclasses
import enum
import json
import pathlib
from collections.abc import Callable
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config


class ExampleType(enum.Enum):
    ALOHA = "aloha"
    DROID = "droid"


@dataclasses.dataclass
class Args:
    config: str = "pi05_aloha"
    checkpoint_dir: str = "/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base"
    example: ExampleType = ExampleType.ALOHA
    camera: str = "base_0_rgb"
    output: str = ""
    include_intermediates: bool = True
    encoder_layer_limit: int = 1
    numpy_seed: int = 0


def make_example(example: ExampleType) -> dict:
    match example:
        case ExampleType.ALOHA:
            return aloha_policy.make_aloha_example()
        case ExampleType.DROID:
            return droid_policy.make_droid_example()


def module_jit_fn(module: nnx.Module, fn: Callable[..., Any]) -> Callable[..., Any]:
    graphdef, state = nnx.split(module)

    @jax.jit
    def wrapped(state, *args):
        model = nnx.merge(graphdef, state)
        return fn(model, *args)

    def call(*args):
        return wrapped(state, *args)

    return call


def make_single_camera_image_encoder_fn(
    model: _model.BaseModel,
    *,
    include_intermediates: bool,
    encoder_layer_limit: int,
) -> Callable[[jnp.ndarray], Any]:
    def encode(model, image):
        tokens, out = model.PaliGemma.img(image, train=False)
        if not include_intermediates:
            return tokens, {}
        selected = {
            "stem": out["stem"],
            "with_posemb": out["with_posemb"],
            "encoded": out["encoded"],
            "pre_logits_2d": out["pre_logits_2d"],
            "logits_2d": out["logits_2d"],
        }
        for layer in range(encoder_layer_limit):
            block = out["encoder"][f"block{layer:02d}"]
            selected[f"encoder_block{layer:02d}_sa"] = block["sa"]
            selected[f"encoder_block{layer:02d}_plus_sa"] = block["+sa"]
            selected[f"encoder_block{layer:02d}_mlp"] = block["mlp"]
            selected[f"encoder_block{layer:02d}_plus_mlp"] = block["+mlp"]
        return tokens, selected

    return module_jit_fn(model, encode)


def as_float32_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float32)


def print_array_summary(name: str, value: np.ndarray) -> None:
    print(
        f"{name} shape={value.shape} dtype={value.dtype} "
        f"min={float(value.min()):.6g} max={float(value.max()):.6g} mean={float(value.mean()):.6g}"
    )


def default_output_path(args: Args) -> pathlib.Path:
    safe_camera = args.camera.replace("/", "_")
    return pathlib.Path("build/openpi_golden") / f"{args.config}_{args.example.value}_{safe_camera}_image_encoder.npz"


def main(args: Args) -> None:
    checkpoint_dir = pathlib.Path(download.maybe_download(args.checkpoint_dir))
    train_config = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    if policy._is_pytorch_model:  # noqa: SLF001
        raise ValueError("This exporter currently supports the JAX policy path only.")

    numpy_state = np.random.get_state()
    try:
        np.random.seed(args.numpy_seed)
        example = make_example(args.example)
    finally:
        np.random.set_state(numpy_state)
    inputs = policy._input_transform(jax.tree.map(lambda x: x, example))  # noqa: SLF001
    batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched_inputs)
    observation = _model.preprocess_observation(None, observation, train=False)

    if args.camera not in observation.images:
        raise ValueError(f"Camera {args.camera!r} not found. Available cameras: {tuple(observation.images)}")

    encode_image = make_single_camera_image_encoder_fn(
        policy._model,  # noqa: SLF001
        include_intermediates=args.include_intermediates,
        encoder_layer_limit=args.encoder_layer_limit,
    )
    image = observation.images[args.camera]
    tokens, intermediates = encode_image(image)
    jax.block_until_ready(tokens)

    image_np = as_float32_array(image)
    tokens_np = as_float32_array(tokens)
    mask_np = np.asarray(jax.device_get(observation.image_masks[args.camera]), dtype=np.bool_)

    output_path = pathlib.Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "example": args.example.value,
        "camera": args.camera,
        "image_dtype_contract": "float32 in [-1, 1], NHWC, after OpenPI input transforms and preprocess_observation",
        "tokens_contract": "decoder-ready image token embeddings from model.PaliGemma.img(image, train=False)",
        "image_shape": list(image_np.shape),
        "tokens_shape": list(tokens_np.shape),
        "available_cameras": list(observation.images),
        "include_intermediates": args.include_intermediates,
        "encoder_layer_limit": args.encoder_layer_limit,
        "numpy_seed": args.numpy_seed,
    }

    arrays: dict[str, Any] = {
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
        "image": image_np,
        "image_mask": mask_np,
        "tokens": tokens_np,
    }
    for name, value in intermediates.items():
        arrays[f"intermediate_{name}"] = as_float32_array(value)

    np.savez_compressed(output_path, **arrays)

    print(f"wrote {output_path}")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print_array_summary("image", image_np)
    print_array_summary("tokens", tokens_np)
    for name in sorted(k for k in arrays if k.startswith("intermediate_")):
        print_array_summary(name, arrays[name])


if __name__ == "__main__":
    main(tyro.cli(Args))
