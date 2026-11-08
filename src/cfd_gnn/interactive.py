from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from cfd_gnn.data import GRID_SIZE, HitDataset
from cfd_gnn.model import load_checkpoint
from cfd_gnn.visualize import predict_snapshot


def interactive_viewer(data_dir: Path, checkpoint: Path) -> None:
    dataset = HitDataset(data_dir, grid_size=GRID_SIZE)
    params, metadata = load_checkpoint(str(checkpoint))
    frame_steps = np.linspace(
        dataset.steps[0], dataset.steps[-2], num=25, dtype=np.int32
    )
    grid = pv.StructuredGrid()
    grid.dimensions = (dataset.size, dataset.size, dataset.size)
    coordinates = [
        np.asarray(axis, dtype=np.float32).reshape(
            (dataset.size, dataset.size, dataset.size), order="C"
        )
        for axis in dataset.coordinates
    ]
    grid.points = np.column_stack(
        [axis.ravel(order="F") for axis in coordinates]
    )

    plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 800))
    plotter.set_background("#f6f8fc")
    for column, title in enumerate(("GNN one-step prediction", "DNS reference")):
        plotter.subplot(0, column)
        plotter.add_text(
            title, position="upper_left", font_size=16, color="#182033"
        )
        plotter.add_text(
            "Contours: Y_O2 = 0.15, 0.30, 0.50",
            position="lower_left",
            font_size=11,
            color="#182033",
        )
        plotter.add_mesh(grid.outline(), color="#313a4b", line_width=1.2)
        plotter.add_axes(line_width=2)
        plotter.view_isometric()
        plotter.camera.parallel_projection = True
        plotter.camera.zoom(1.15)
    plotter.link_views()

    playing = False

    def render_frame(index: int) -> None:
        index = max(0, min(index, len(frame_steps) - 1))
        step = int(frame_steps[index])
        prediction, reference = predict_snapshot(dataset, params, metadata, step)
        for column, field in enumerate((prediction[..., 6], reference[..., 6])):
            plotter.subplot(0, column)
            grid.point_data["oxygen_mass_fraction"] = field.ravel(order="F")
            surface = grid.contour(
                isosurfaces=[0.15, 0.30, 0.50],
                scalars="oxygen_mass_fraction",
            )
            plotter.add_mesh(
                surface,
                scalars="oxygen_mass_fraction",
                cmap="plasma",
                clim=(0.15, 0.50),
                opacity=0.82,
                smooth_shading=True,
                show_scalar_bar=False,
                name=f"oxygen_surface_{column}",
            )
        plotter.add_text(
            f"DNS input {step:03d} -> target {step + 1:03d}",
            position="upper_right",
            font_size=13,
            color="#182033",
            name="step_label",
        )
        plotter.render()

    def on_slider(value: float) -> None:
        render_frame(round(value))

    def step_frame(offset: int) -> None:
        current = int(round(slider.GetRepresentation().GetValue()))
        next_index = (current + offset) % len(frame_steps)
        slider.GetRepresentation().SetValue(next_index)
        render_frame(next_index)

    def toggle_playback() -> None:
        nonlocal playing
        playing = not playing

    def on_timer(_step: int) -> None:
        if playing:
            step_frame(1)

    render_frame(0)
    slider = plotter.add_slider_widget(
        on_slider,
        (0, len(frame_steps) - 1),
        value=0,
        title="DNS input snapshot",
        pointa=(0.32, 0.93),
        pointb=(0.70, 0.93),
        interaction_event="end",
    )
    plotter.add_text(
        "Left / Right: step    Space: play/pause    Rotate: drag",
        position="lower_right",
        font_size=11,
        color="#182033",
    )
    plotter.add_key_event("Left", lambda: step_frame(-1))
    plotter.add_key_event("Right", lambda: step_frame(1))
    plotter.add_key_event("space", toggle_playback)
    plotter.add_timer_event(max_steps=100_000, duration=700, callback=on_timer)
    plotter.show(
        title="PyVista: GNN prediction vs DNS",
        window_size=(1400, 800),
        interactive=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively compare the GNN oxygen-core prediction with DNS"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/hit_gnn.npz"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    interactive_viewer(args.data_dir, args.checkpoint)


if __name__ == "__main__":
    main()
