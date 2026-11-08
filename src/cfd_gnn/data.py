from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

VARIABLES = (
    "UX_ms-1",
    "UY_ms-1",
    "UZ_ms-1",
    "P_Pa",
    "T_K",
    "RHO_kgm-3",
    "YO2",
    "YCH4",
)
GRID_FILES = ("X_m.dat", "Y_m.dat", "Z_m.dat")
GRID_SIZE = 129
CHANNELS = len(VARIABLES)


def read_float_grid(path: Path, size: int = GRID_SIZE) -> np.ndarray:
    expected_values = size**3
    expected_bytes = expected_values * np.dtype("<f4").itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {actual_bytes} bytes; expected {expected_bytes} "
            f"for a {size}x{size}x{size} float32 grid"
        )
    return np.memmap(path, dtype="<f4", mode="r", shape=(expected_values,))


def periodic_indices(start: int, length: int, size: int, halo: int) -> np.ndarray:
    return np.arange(start - halo, start + length + halo, dtype=np.int64) % size


def flat_patch_indices(
    starts: Sequence[int],
    lengths: Sequence[int],
    size: int,
    halo: int,
) -> np.ndarray:
    axes = [
        periodic_indices(start, length, size, halo)
        for start, length in zip(starts, lengths, strict=True)
    ]
    return (
        (axes[0][:, None, None] * size + axes[1][None, :, None]) * size
        + axes[2][None, None, :]
    )


def extract_patch(
    flat_array: np.ndarray,
    starts: Sequence[int],
    lengths: Sequence[int],
    size: int,
    halo: int,
) -> np.ndarray:
    indices = flat_patch_indices(starts, lengths, size, halo)
    return np.asarray(flat_array[indices], dtype=np.float32)


class HitDataset:
    """Memory-mapped access to the Kaggle HIT snapshots and periodic grid."""

    def __init__(self, root: Path, grid_size: int = GRID_SIZE) -> None:
        if grid_size < 1:
            raise ValueError("Grid size must be positive")
        self.root = Path(root)
        self.size = grid_size
        self.node_count = grid_size**3
        self.data_dir = self.root / "data"
        self.grid_dir = self.root / "grid"
        self.steps = self._find_steps()
        self.transitions = tuple(zip(self.steps[:-1], self.steps[1:], strict=True))
        if not self.transitions:
            raise ValueError(f"No consecutive snapshots found in {self.data_dir}")
        self.coordinates = tuple(
            read_float_grid(self.grid_dir / name, grid_size) for name in GRID_FILES
        )

    def _find_steps(self) -> tuple[int, ...]:
        present: dict[str, set[int]] = {}
        for variable in VARIABLES:
            steps = {
                int(path.stem.rsplit("_id", 1)[1])
                for path in self.data_dir.glob(f"{variable}_id*.dat")
                if "_id" in path.stem and path.stem.rsplit("_id", 1)[1].isdigit()
            }
            if not steps:
                raise FileNotFoundError(
                    f"No {variable} snapshots found in {self.data_dir}. "
                    "Download and extract the Kaggle dataset first."
                )
            present[variable] = steps

        reference = present[VARIABLES[0]]
        for variable, steps in present.items():
            if steps != reference:
                missing = sorted(reference - steps)
                extra = sorted(steps - reference)
                raise ValueError(
                    f"Snapshot IDs for {variable} differ from {VARIABLES[0]} "
                    f"(missing={missing[:5]}, extra={extra[:5]})"
                )

        ordered = tuple(sorted(reference))
        expected = tuple(range(ordered[0], ordered[-1] + 1))
        if ordered != expected:
            raise ValueError("Snapshot IDs must be contiguous for next-step training")
        return ordered

    @lru_cache(maxsize=4)
    def _snapshot(self, step: int) -> tuple[np.ndarray, ...]:
        if step not in self.steps:
            raise ValueError(f"Unknown snapshot ID {step}")
        return tuple(
            read_float_grid(
                self.data_dir / f"{variable}_id{step:03d}.dat", self.size
            )
            for variable in VARIABLES
        )

    def state_patch(
        self,
        step: int,
        starts: Sequence[int],
        lengths: Sequence[int],
        halo: int,
    ) -> np.ndarray:
        indices = flat_patch_indices(starts, lengths, self.size, halo)
        return np.stack(
            [
                np.asarray(field[indices], dtype=np.float32)
                for field in self._snapshot(step)
            ],
            axis=-1,
        )

    def state_grid(self, step: int) -> np.ndarray:
        return np.stack(
            [
                np.asarray(field).reshape(
                    (self.size, self.size, self.size), order="C"
                )
                for field in self._snapshot(step)
            ],
            axis=-1,
        )

    def coordinate_patch(
        self,
        starts: Sequence[int],
        lengths: Sequence[int],
        halo: int,
        coordinate_scale: float,
    ) -> np.ndarray:
        indices = flat_patch_indices(starts, lengths, self.size, halo)
        return np.stack(
            [
                np.asarray(coordinate[indices], dtype=np.float32) / coordinate_scale
                for coordinate in self.coordinates
            ],
            axis=-1,
        )

    def normalization_stats(
        self, training_steps: Sequence[int], sample_count: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:
        if not training_steps:
            raise ValueError("At least one training snapshot is required")
        if sample_count < 1:
            raise ValueError("Normalization sample count must be positive")
        sample_count = min(sample_count, self.node_count)
        sample_ids = np.linspace(
            0, self.node_count - 1, num=sample_count, dtype=np.int64
        )
        samples: list[list[np.ndarray]] = [[] for _ in VARIABLES]
        for step in training_steps:
            for channel, field in enumerate(self._snapshot(step)):
                samples[channel].append(np.asarray(field[sample_ids], dtype=np.float64))

        means = np.array(
            [np.concatenate(channel).mean() for channel in samples], dtype=np.float32
        )
        standard_deviations = np.array(
            [np.concatenate(channel).std() for channel in samples], dtype=np.float32
        )
        standard_deviations = np.maximum(standard_deviations, 1e-6)
        return means, standard_deviations

    def coordinate_scale(self) -> float:
        scale = max(float(np.max(np.abs(axis))) for axis in self.coordinates)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Grid coordinates must have a finite nonzero scale")
        return scale
