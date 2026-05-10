import dataclasses
import enum
import pathlib
import statistics
import time
from collections.abc import Callable
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.shared import gemma_cpp_image_encoder_bridge
from openpi.training import config as _config


class ExampleType(enum.Enum):
    ALOHA = "aloha"
    DROID = "droid"


@dataclasses.dataclass
class Args:
    config: str = "pi05_aloha"
    checkpoint_dir: str = "/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base"
    example: ExampleType = ExampleType.ALOHA
    warmup: int = 3
    runs: int = 10
    num_steps: int = 10
    external_image_tokens: str = ""


def make_example(example: ExampleType) -> dict:
    match example:
        case ExampleType.ALOHA:
            return aloha_policy.make_aloha_example()
        case ExampleType.DROID:
            return droid_policy.make_droid_example()


def block_until_ready(value):
    return jax.block_until_ready(value)


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    block_until_ready(value)
    return value, time.perf_counter() - start


def summarize(name: str, values: list[float]) -> None:
    sorted_values = sorted(values)
    avg = statistics.mean(values)
    median = statistics.median(values)
    p90 = sorted_values[max(0, min(len(sorted_values) - 1, int(len(sorted_values) * 0.9) - 1))]
    print(f"{name}_avg_sec {avg}")
    print(f"{name}_median_sec {median}")
    print(f"{name}_p90_sec {p90}")
    print(f"{name}_min_sec {sorted_values[0]}")
    print(f"{name}_max_sec {sorted_values[-1]}")


def module_jit_fn(module: nnx.Module, fn: Callable[..., Any]) -> Callable[..., Any]:
    graphdef, state = nnx.split(module)

    @jax.jit
    def wrapped(state, *args):
        model = nnx.merge(graphdef, state)
        return fn(model, *args)

    def call(*args):
        return wrapped(state, *args)

    return call


def make_image_embed_fn(model: _model.BaseModel) -> Callable[[_model.Observation], Any]:
    def image_embed(model, observation):
        observation = _model.preprocess_observation(None, observation, train=False)
        input_mask = []
        tokens = []
        ar_mask = []
        for name in observation.images:
            image_tokens, _ = model.PaliGemma.img(observation.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                jnp.broadcast_to(
                    observation.image_masks[name][:, None],
                    (observation.image_masks[name].shape[0], image_tokens.shape[1]),
                )
            )
            ar_mask.append(jnp.zeros((image_tokens.shape[1],), dtype=jnp.bool_))
        return (
            observation,
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask),
        )

    return module_jit_fn(model, image_embed)


def make_external_image_embed_fn(
    model: _model.BaseModel, image_tokens: jax.Array
) -> Callable[[_model.Observation], Any]:
    def image_embed(_model_instance, observation):
        observation = _model.preprocess_observation(None, observation, train=False)
        input_mask = []
        tokens = []
        ar_mask = []
        for name in observation.images:
            repeated_tokens = jnp.broadcast_to(
                image_tokens,
                (
                    observation.image_masks[name].shape[0],
                    image_tokens.shape[1],
                    image_tokens.shape[2],
                ),
            )
            tokens.append(repeated_tokens)
            input_mask.append(
                jnp.broadcast_to(
                    observation.image_masks[name][:, None],
                    (observation.image_masks[name].shape[0], repeated_tokens.shape[1]),
                )
            )
            ar_mask.append(jnp.zeros((repeated_tokens.shape[1],), dtype=jnp.bool_))
        return (
            observation,
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask),
        )

    return module_jit_fn(model, image_embed)


def make_text_prefix_fn(model: _model.BaseModel) -> Callable[..., Any]:
    def text_prefix(model, observation, image_tokens, image_mask, image_ar_mask):
        if observation.tokenized_prompt is None:
            return observation, image_tokens, image_mask, image_ar_mask
        tokenized_inputs = model.PaliGemma.llm(observation.tokenized_prompt, method="embed")
        prefix_tokens = jnp.concatenate([image_tokens, tokenized_inputs], axis=1)
        prefix_mask = jnp.concatenate([image_mask, observation.tokenized_prompt_mask], axis=1)
        text_ar_mask = jnp.zeros((tokenized_inputs.shape[1],), dtype=jnp.bool_)
        prefix_ar_mask = jnp.concatenate([image_ar_mask, text_ar_mask])
        return observation, prefix_tokens, prefix_mask, prefix_ar_mask

    return module_jit_fn(model, text_prefix)


def make_prefix_llm_fn(model: _model.BaseModel) -> Callable[..., Any]:
    def prefix_llm(model, observation, prefix_tokens, prefix_mask, prefix_ar_mask):
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return observation, prefix_tokens, prefix_mask, kv_cache

    return module_jit_fn(model, prefix_llm)


def make_suffix_loop_fn(model: _model.BaseModel, num_steps: int) -> Callable[..., Any]:
    def suffix_loop(model, observation, prefix_tokens, prefix_mask, kv_cache, noise):
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time_value = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time_value, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = jnp.broadcast_to(
                prefix_mask[:, None, :],
                (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1]),
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            prefix_out, suffix_out = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )[0]
            del prefix_out
            v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
            return x_t + dt * v_t, time_value + dt

        def cond(carry):
            _, time_value = carry
            return time_value >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions

    return module_jit_fn(model, suffix_loop)


def main(args: Args) -> None:
    checkpoint_dir = download.maybe_download(args.checkpoint_dir)
    train_config = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": args.num_steps},
    )
    if policy._is_pytorch_model:  # noqa: SLF001
        raise ValueError("This breakdown script currently supports the JAX policy path only.")

    if args.external_image_tokens:
        external_tokens = gemma_cpp_image_encoder_bridge.load_tokens_numpy(pathlib.Path(args.external_image_tokens))
        external_tokens = jnp.asarray(external_tokens, dtype=jnp.dtype(train_config.model.dtype))
        image_embed = make_external_image_embed_fn(policy._model, external_tokens)  # noqa: SLF001
        image_embed_source = pathlib.Path(args.external_image_tokens)
    else:
        image_embed = make_image_embed_fn(policy._model)  # noqa: SLF001
        image_embed_source = "openpi_jax"
    text_prefix = make_text_prefix_fn(policy._model)  # noqa: SLF001
    prefix_llm = make_prefix_llm_fn(policy._model)  # noqa: SLF001
    suffix_loop = make_suffix_loop_fn(policy._model, args.num_steps)  # noqa: SLF001

    example = make_example(args.example)
    inputs = policy._input_transform(jax.tree.map(lambda x: x, example))  # noqa: SLF001
    batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(batched_inputs)
    rng = jax.random.key(0)
    noise = jax.random.normal(
        rng,
        (1, train_config.model.action_horizon, train_config.model.action_dim),
    )

    print(f"config {args.config}")
    print(f"checkpoint_dir {pathlib.Path(checkpoint_dir)}")
    print(f"example {args.example.value}")
    print(f"num_steps {args.num_steps}")
    print(f"image_embed_source {image_embed_source}")
    print(f"action_horizon {train_config.model.action_horizon}")
    print(f"action_dim {train_config.model.action_dim}")

    # Compile and warm device paths.
    image_embedded = image_embed(observation)
    prefix_built = text_prefix(*image_embedded)
    prefilled = prefix_llm(*prefix_built)
    block_until_ready(prefilled)
    actions = suffix_loop(*prefilled, noise)
    block_until_ready(actions)
    policy.infer(example)

    for _ in range(args.warmup):
        image_embedded = image_embed(observation)
        prefix_built = text_prefix(*image_embedded)
        prefilled = prefix_llm(*prefix_built)
        actions = suffix_loop(*prefilled, noise)
        block_until_ready(actions)
        policy.infer(example)

    total_times: list[float] = []
    input_transform_times: list[float] = []
    to_device_times: list[float] = []
    image_embed_times: list[float] = []
    text_prefix_times: list[float] = []
    prefix_llm_times: list[float] = []
    prefix_total_times: list[float] = []
    suffix_loop_times: list[float] = []
    output_transform_times: list[float] = []

    for _ in range(args.runs):
        _, elapsed = time_call(lambda: policy.infer(example))
        total_times.append(elapsed)

        transformed, elapsed = time_call(
            lambda: policy._input_transform(jax.tree.map(lambda x: x, example))  # noqa: SLF001
        )
        input_transform_times.append(elapsed)

        def to_observation():
            batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
            return _model.Observation.from_dict(batched)

        obs, elapsed = time_call(to_observation)
        to_device_times.append(elapsed)

        image_embedded, elapsed = time_call(lambda: image_embed(obs))
        image_embed_times.append(elapsed)

        prefix_built, elapsed = time_call(lambda: text_prefix(*image_embedded))
        text_prefix_times.append(elapsed)

        prefilled, elapsed = time_call(lambda: prefix_llm(*prefix_built))
        prefix_llm_times.append(elapsed)

        prefix_total_times.append(image_embed_times[-1] + text_prefix_times[-1] + prefix_llm_times[-1])

        actions, elapsed = time_call(lambda: suffix_loop(*prefilled, noise))
        suffix_loop_times.append(elapsed)

        def transform_outputs():
            outputs = {
                "state": np.asarray(obs.state[0, ...]),
                "actions": np.asarray(actions[0, ...]),
            }
            return policy._output_transform(outputs)  # noqa: SLF001

        _, elapsed = time_call(transform_outputs)
        output_transform_times.append(elapsed)

    summarize("total_policy_infer", total_times)
    summarize("input_transform", input_transform_times)
    summarize("to_device_observation", to_device_times)
    summarize("image_embed", image_embed_times)
    summarize("text_prefix", text_prefix_times)
    summarize("prefix_llm", prefix_llm_times)
    summarize("prefix_total_composed", prefix_total_times)
    summarize("suffix_loop", suffix_loop_times)
    summarize("output_transform", output_transform_times)

    print("first_total_sec", [round(x, 6) for x in total_times[:10]])
    print("first_image_embed_sec", [round(x, 6) for x in image_embed_times[:10]])
    print("first_prefix_llm_sec", [round(x, 6) for x in prefix_llm_times[:10]])
    print("first_prefix_total_composed_sec", [round(x, 6) for x in prefix_total_times[:10]])
    print("first_suffix_loop_sec", [round(x, 6) for x in suffix_loop_times[:10]])


if __name__ == "__main__":
    main(tyro.cli(Args))
