import numpy as np
import jax

from cfd_gnn.data import GRID_FILES, VARIABLES, HitDataset
from cfd_gnn.model import initialize_parameters
from cfd_gnn.visualize import _grid, predict_snapshot


def test_pyvista_grid_points_follow_binary_coordinate_axes(tmp_path):
    size = 2
    data_dir = tmp_path / "data"
    grid_dir = tmp_path / "grid"
    data_dir.mkdir()
    grid_dir.mkdir()
    coordinates = np.meshgrid(
        np.arange(size, dtype=np.float32),
        np.arange(size, dtype=np.float32) + 10,
        np.arange(size, dtype=np.float32) + 20,
        indexing="ij",
    )
    for filename, coordinate in zip(GRID_FILES, coordinates, strict=True):
        (grid_dir / filename).write_bytes(
            coordinate.ravel(order="C").astype("<f4").tobytes()
        )
    for step in (0, 1):
        for variable in VARIABLES:
            values = np.zeros(size**3, dtype="<f4")
            (data_dir / f"{variable}_id{step:03d}.dat").write_bytes(values.tobytes())

    dataset = HitDataset(tmp_path, grid_size=size)
    grid = _grid(dataset)
    point_id = np.ravel_multi_index((1, 0, 0), (size, size, size), order="F")

    assert grid.dimensions == (size, size, size)
    np.testing.assert_array_equal(grid.points[point_id], [1, 10, 20])

    params = initialize_parameters(jax.random.PRNGKey(0), 11, 4, 8, 1)
    prediction, reference = predict_snapshot(
        dataset,
        params,
        {
            "mean": np.zeros(8, dtype=np.float32),
            "standard_deviation": np.ones(8, dtype=np.float32),
            "coordinate_scale": 1.0,
            "patch_size": 1,
            "layers": 1,
        },
        step=0,
    )
    assert prediction.shape == (size, size, size, 8)
    np.testing.assert_array_equal(reference, np.zeros_like(reference))
