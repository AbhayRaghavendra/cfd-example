import numpy as np
import pytest

from cfd_gnn.data import (
    GRID_FILES,
    VARIABLES,
    HitDataset,
    extract_patch,
    flat_patch_indices,
    read_float_grid,
)


def test_periodic_patch_wraps_each_axis():
    grid = np.arange(4**3, dtype=np.float32).reshape(4, 4, 4)
    flat = grid.ravel(order="C")

    patch = extract_patch(flat, (0, 1, 3), (2, 2, 2), size=4, halo=1)

    expected = grid[np.ix_([3, 0, 1, 2], [0, 1, 2, 3], [2, 3, 0, 1])]
    np.testing.assert_array_equal(patch, expected)


def test_flat_patch_indices_match_c_order():
    indices = flat_patch_indices((1, 2, 3), (1, 1, 1), size=4, halo=0)

    assert indices.shape == (1, 1, 1)
    assert indices.item() == (1 * 4 + 2) * 4 + 3


def test_read_float_grid_rejects_unexpected_file_size(tmp_path):
    path = tmp_path / "short.dat"
    path.write_bytes(np.zeros(7, dtype="<f4").tobytes())

    with pytest.raises(ValueError, match="expected"):
        read_float_grid(path, size=2)


def test_hit_dataset_loads_binary_snapshots_and_periodic_coordinates(tmp_path):
    size = 2
    data_dir = tmp_path / "data"
    grid_dir = tmp_path / "grid"
    data_dir.mkdir()
    grid_dir.mkdir()
    coordinates = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32),
        indexing="ij",
    )
    for path, coordinate in zip(GRID_FILES, coordinates, strict=True):
        (grid_dir / path).write_bytes(
            coordinate.ravel(order="C").astype("<f4").tobytes()
        )
    for step in (0, 1):
        for channel, variable in enumerate(VARIABLES):
            values = np.full((size, size, size), step + channel, dtype="<f4")
            (data_dir / f"{variable}_id{step:03d}.dat").write_bytes(
                values.ravel(order="C").tobytes()
            )

    dataset = HitDataset(tmp_path, grid_size=size)
    patch = dataset.state_patch(0, (0, 0, 0), (1, 1, 1), halo=0)

    assert dataset.steps == (0, 1)
    assert dataset.transitions == ((0, 1),)
    np.testing.assert_array_equal(patch[0, 0, 0], np.arange(len(VARIABLES)))
    np.testing.assert_array_equal(dataset.state_grid(1)[0, 0, 0], np.arange(1, 9))
