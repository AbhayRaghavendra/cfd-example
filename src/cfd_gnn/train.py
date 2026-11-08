from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cfd_gnn.data import CHANNELS, HitDataset
from cfd_gnn.model import apply_model, initialize_parameters, save_checkpoint


def _training_functions(halo: int):
    @jax.jit
    def train_step(params, optimizer_state, features, targets, learning_rate):
        first_moment, second_moment, step = optimizer_state

        def loss_function(current_params):
            prediction = apply_model(current_params, features)
            core_prediction = prediction[
                :, halo:-halo, halo:-halo, halo:-halo, :
            ]
            core_targets = targets[:, halo:-halo, halo:-halo, halo:-halo, :]
            return jnp.mean(jnp.square(core_prediction - core_targets))

        loss, gradients = jax.value_and_grad(loss_function)(params)
        step = step + 1
        first_moment = jax.tree_util.tree_map(
            lambda moment, gradient: 0.9 * moment + 0.1 * gradient,
            first_moment,
            gradients,
        )
        second_moment = jax.tree_util.tree_map(
            lambda moment, gradient: 0.999 * moment + 0.001 * jnp.square(gradient),
            second_moment,
            gradients,
        )
        corrected_first = jax.tree_util.tree_map(
            lambda moment: moment / (1.0 - jnp.power(0.9, step)),
            first_moment,
        )
        corrected_second = jax.tree_util.tree_map(
            lambda moment: moment / (1.0 - jnp.power(0.999, step)),
            second_moment,
        )
        params = jax.tree_util.tree_map(
            lambda parameter, first, second: parameter
            - learning_rate * first / (jnp.sqrt(second) + 1e-8),
            params,
            corrected_first,
            corrected_second,
        )
        return params, (first_moment, second_moment, step), loss

    @jax.jit
    def evaluate(params, features, targets):
        prediction = apply_model(params, features)
        core_prediction = prediction[:, halo:-halo, halo:-halo, halo:-halo, :]
        core_targets = targets[:, halo:-halo, halo:-halo, halo:-halo, :]
        return jnp.mean(jnp.square(core_prediction - core_targets))

    return train_step, evaluate


def _batch(
    dataset: HitDataset,
    transitions: tuple[tuple[int, int], ...],
    starts: list[tuple[int, int, int]],
    patch_size: int,
    halo: int,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    coordinate_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for transition, start in zip(transitions, starts, strict=True):
        previous, following = transition
        state = dataset.state_patch(
            previous, start, (patch_size, patch_size, patch_size), halo
        )
        target = dataset.state_patch(
            following, start, (patch_size, patch_size, patch_size), halo
        )
        coordinates = dataset.coordinate_patch(
            start,
            (patch_size, patch_size, patch_size),
            halo,
            coordinate_scale,
        )
        features.append(
            np.concatenate(
                ((state - mean) / standard_deviation, coordinates), axis=-1
            )
        )
        targets.append((target - mean) / standard_deviation)
    return np.stack(features), np.stack(targets)


def _sample_starts(
    rng: np.random.Generator, count: int, grid_size: int
) -> list[tuple[int, int, int]]:
    return [
        tuple(
            int(rng.integers(0, grid_size))
            for _ in range(3)
        )
        for _ in range(count)
    ]


def train(args: argparse.Namespace) -> Path:
    if args.patch_size < 1:
        raise ValueError("--patch-size must be positive")
    if args.layers < 1:
        raise ValueError("--layers must be positive")
    if args.batch_size < 1 or args.samples_per_epoch < 1 or args.epochs < 1:
        raise ValueError("epochs, batch size, and samples per epoch must be positive")

    dataset = HitDataset(args.data_dir, grid_size=args.grid_size)
    if args.patch_size > dataset.size:
        raise ValueError("--patch-size cannot exceed the grid size")
    if args.validation_samples < 1:
        raise ValueError("--validation-samples must be positive")
    transitions = dataset.transitions
    split = max(1, int(0.8 * len(transitions)))
    train_transitions = transitions[:split]
    validation_transitions = transitions[split:]
    if not validation_transitions:
        raise ValueError("At least two temporal transitions are required")

    train_steps = tuple(
        sorted({step for pair in train_transitions for step in pair})
    )
    mean, standard_deviation = dataset.normalization_stats(train_steps)
    coordinate_scale = dataset.coordinate_scale()
    halo = args.layers
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    params = initialize_parameters(
        key, CHANNELS + 3, args.hidden_size, CHANNELS, args.layers
    )
    optimizer_state = (
        jax.tree_util.tree_map(jnp.zeros_like, params),
        jax.tree_util.tree_map(jnp.zeros_like, params),
        jnp.asarray(0, dtype=jnp.int32),
    )
    train_step, evaluate = _training_functions(halo)
    validation_count = min(args.validation_samples, len(validation_transitions))
    validation_indices = np.linspace(
        0, len(validation_transitions) - 1, validation_count, dtype=np.int64
    )
    validation_batch = _batch(
        dataset,
        tuple(validation_transitions[index] for index in validation_indices),
        _sample_starts(rng, validation_count, dataset.size),
        args.patch_size,
        halo,
        mean,
        standard_deviation,
        coordinate_scale,
    )
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        for offset in range(0, args.samples_per_epoch, args.batch_size):
            batch_count = min(args.batch_size, args.samples_per_epoch - offset)
            transition_indices = rng.integers(
                0, len(train_transitions), size=batch_count
            )
            batch_transitions = tuple(
                train_transitions[index] for index in transition_indices
            )
            features, targets = _batch(
                dataset,
                batch_transitions,
                _sample_starts(rng, batch_count, dataset.size),
                args.patch_size,
                halo,
                mean,
                standard_deviation,
                coordinate_scale,
            )
            params, optimizer_state, loss = train_step(
                params,
                optimizer_state,
                features,
                targets,
                args.learning_rate,
            )
            epoch_losses.append(float(loss))

        validation_loss = float(evaluate(params, *validation_batch))
        train_loss = float(np.mean(epoch_losses))
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_mse={train_loss:.6g} val_mse={validation_loss:.6g}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(
                str(output),
                params,
                mean,
                standard_deviation,
                coordinate_scale,
                args.patch_size,
                args.hidden_size,
                args.layers,
                int(optimizer_state[2]),
            )

    print(f"Best checkpoint: {output} (validation MSE {best_loss:.6g})")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a periodic message-passing GNN on BLASTNet HIT"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--grid-size", type=int, default=129, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/hit_gnn.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    train(_parser().parse_args())


if __name__ == "__main__":
    main()
