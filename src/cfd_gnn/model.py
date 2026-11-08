from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Params = dict[str, Any]


def _dense_parameters(
    key: jax.Array, input_size: int, output_size: int
) -> tuple[jax.Array, jax.Array]:
    limit = jnp.sqrt(6.0 / (input_size + output_size))
    weight = jax.random.uniform(
        key, (input_size, output_size), minval=-limit, maxval=limit
    )
    return weight, jnp.zeros((output_size,), dtype=jnp.float32)


def initialize_parameters(
    key: jax.Array, input_size: int, hidden_size: int, output_size: int, layers: int
) -> Params:
    keys = jax.random.split(key, layers + 2)
    encoder_weight, encoder_bias = _dense_parameters(keys[0], input_size, hidden_size)
    message_parameters = [
        _dense_parameters(keys[index + 1], 2 * hidden_size, hidden_size)
        for index in range(layers)
    ]
    decoder_weight, decoder_bias = _dense_parameters(
        keys[-1], hidden_size, output_size
    )
    return {
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "message_weights": tuple(item[0] for item in message_parameters),
        "message_biases": tuple(item[1] for item in message_parameters),
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
    }


def apply_model(params: Params, features: jax.Array) -> jax.Array:
    hidden = jnp.tanh(features @ params["encoder_weight"] + params["encoder_bias"])
    for weight, bias in zip(
        params["message_weights"], params["message_biases"], strict=True
    ):
        neighbor_mean = sum(
            jnp.roll(hidden, shift, axis=axis)
            for axis in (1, 2, 3)
            for shift in (-1, 1)
        ) / 6.0
        message = jnp.concatenate((hidden, neighbor_mean - hidden), axis=-1)
        hidden = hidden + 0.5 * jnp.tanh(message @ weight + bias)
    return hidden @ params["decoder_weight"] + params["decoder_bias"]


def save_checkpoint(
    path: str,
    params: Params,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    coordinate_scale: float,
    patch_size: int,
    hidden_size: int,
    layers: int,
    step: int,
) -> None:
    values: dict[str, np.ndarray] = {
        "mean": np.asarray(mean, dtype=np.float32),
        "standard_deviation": np.asarray(standard_deviation, dtype=np.float32),
        "coordinate_scale": np.asarray(coordinate_scale, dtype=np.float32),
        "patch_size": np.asarray(patch_size, dtype=np.int32),
        "hidden_size": np.asarray(hidden_size, dtype=np.int32),
        "layers": np.asarray(layers, dtype=np.int32),
        "step": np.asarray(step, dtype=np.int32),
        "encoder_weight": np.asarray(params["encoder_weight"]),
        "encoder_bias": np.asarray(params["encoder_bias"]),
        "decoder_weight": np.asarray(params["decoder_weight"]),
        "decoder_bias": np.asarray(params["decoder_bias"]),
    }
    values.update(
        {
            f"message_weight_{index}": np.asarray(weight)
            for index, weight in enumerate(params["message_weights"])
        }
    )
    values.update(
        {
            f"message_bias_{index}": np.asarray(bias)
            for index, bias in enumerate(params["message_biases"])
        }
    )
    np.savez_compressed(path, **values)


def load_checkpoint(path: str) -> tuple[Params, dict[str, np.ndarray | float | int]]:
    with np.load(path, allow_pickle=False) as checkpoint:
        layers = int(checkpoint["layers"])
        params: Params = {
            "encoder_weight": jnp.asarray(checkpoint["encoder_weight"]),
            "encoder_bias": jnp.asarray(checkpoint["encoder_bias"]),
            "message_weights": tuple(
                jnp.asarray(checkpoint[f"message_weight_{index}"])
                for index in range(layers)
            ),
            "message_biases": tuple(
                jnp.asarray(checkpoint[f"message_bias_{index}"])
                for index in range(layers)
            ),
            "decoder_weight": jnp.asarray(checkpoint["decoder_weight"]),
            "decoder_bias": jnp.asarray(checkpoint["decoder_bias"]),
        }
        metadata: dict[str, np.ndarray | float | int] = {
            "mean": checkpoint["mean"].copy(),
            "standard_deviation": checkpoint["standard_deviation"].copy(),
            "coordinate_scale": float(checkpoint["coordinate_scale"]),
            "patch_size": int(checkpoint["patch_size"]),
            "hidden_size": int(checkpoint["hidden_size"]),
            "layers": layers,
            "step": int(checkpoint["step"]),
        }
    return params, metadata
