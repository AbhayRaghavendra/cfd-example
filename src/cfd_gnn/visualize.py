from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
import pyvista as pv

from cfd_gnn.data import CHANNELS, GRID_SIZE, HitDataset
from cfd_gnn.model import apply_model, load_checkpoint


def _grid(dataset: HitDataset) -> pv.StructuredGrid:
    coordinates = [
        np.asarray(axis, dtype=np.float32).reshape(
            (dataset.size, dataset.size, dataset.size), order="C"
        )
        for axis in dataset.coordinates
    ]
    grid = pv.StructuredGrid()
    grid.dimensions = (dataset.size, dataset.size, dataset.size)
    grid.points = np.column_stack([axis.ravel(order="F") for axis in coordinates])
    return grid


def predict_snapshot(
    dataset: HitDataset,
    params,
    metadata: dict[str, np.ndarray | float | int],
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if step not in dataset.steps[:-1]:
        raise ValueError(f"Step {step} has no following snapshot")
    patch_size = int(metadata["patch_size"])
    layers = int(metadata["layers"])
    halo = layers
    mean = np.asarray(metadata["mean"])
    standard_deviation = np.asarray(metadata["standard_deviation"])
    coordinate_scale = float(metadata["coordinate_scale"])
    apply = jax.jit(apply_model)
    prediction = np.empty(
        (dataset.size, dataset.size, dataset.size, CHANNELS), dtype=np.float32
    )

    for x_start in range(0, dataset.size, patch_size):
        for y_start in range(0, dataset.size, patch_size):
            for z_start in range(0, dataset.size, patch_size):
                starts = (x_start, y_start, z_start)
                state = dataset.state_patch(
                    step,
                    starts,
                    (patch_size, patch_size, patch_size),
                    halo,
                )
                coordinates = dataset.coordinate_patch(
                    starts,
                    (patch_size, patch_size, patch_size),
                    halo,
                    coordinate_scale,
                )
                features = np.concatenate(
                    ((state - mean) / standard_deviation, coordinates), axis=-1
                )[None, ...]
                normalized = np.asarray(apply(params, features))[0]
                predicted_state = (
                    normalized[
                        halo : halo + patch_size,
                        halo : halo + patch_size,
                        halo : halo + patch_size,
                    ]
                    * standard_deviation
                    + mean
                )
                indices = [
                    (start + np.arange(patch_size)) % dataset.size
                    for start in starts
                ]
                prediction[np.ix_(*indices)] = predicted_state

    target = dataset.state_grid(step + 1)
    return prediction, target


def visualize(args: argparse.Namespace) -> Path:
    dataset = HitDataset(args.data_dir, grid_size=args.grid_size)
    params, metadata = load_checkpoint(str(args.checkpoint))
    prediction, target = predict_snapshot(dataset, params, metadata, args.step)
    grid = _grid(dataset)
    field_names = (
        "u",
        "v",
        "w",
        "pressure",
        "temperature",
        "density",
        "oxygen",
        "methane",
    )
    for channel, name in enumerate(field_names):
        grid.point_data[f"predicted_{name}"] = prediction[..., channel].ravel(
            order="F"
        )
        grid.point_data[f"reference_{name}"] = target[..., channel].ravel(order="F")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    if args.screenshot is not None:
        Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(
            grid.slice(normal="z"),
            scalars=args.field,
            show_edges=False,
        )
        plotter.add_axes()
        plotter.screenshot(args.screenshot)
        plotter.close()
    print(f"Saved PyVista structured-grid prediction: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict and visualize a BLASTNet HIT snapshot"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--grid-size", type=int, default=GRID_SIZE, help=argparse.SUPPRESS
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/hit_gnn.npz"))
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prediction.vts"))
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--field", default="predicted_density")
    return parser


def main() -> None:
    visualize(_parser().parse_args())


if __name__ == "__main__":
    main()
