from argparse import Namespace

import numpy as np

from cfd_gnn.data import GRID_FILES, VARIABLES
from cfd_gnn.model import load_checkpoint
from cfd_gnn.train import train


def test_training_runs_on_small_periodic_dataset(tmp_path):
    size = 8
    data_dir = tmp_path / "raw"
    grid_dir = data_dir / "grid"
    field_dir = data_dir / "data"
    grid_dir.mkdir(parents=True)
    field_dir.mkdir()
    coordinates = np.meshgrid(
        *[np.linspace(-1, 1, size, dtype=np.float32) for _ in range(3)],
        indexing="ij",
    )
    for filename, coordinate in zip(GRID_FILES, coordinates, strict=True):
        (grid_dir / filename).write_bytes(
            coordinate.ravel(order="C").astype("<f4").tobytes()
        )

    spatial_pattern = (
        np.arange(size**3, dtype=np.float32).reshape(size, size, size) / size**3
    )
    for step in range(3):
        for channel, variable in enumerate(VARIABLES):
            field = channel + (step + 1) * spatial_pattern
            (field_dir / f"{variable}_id{step:03d}.dat").write_bytes(
                field.astype("<f4").ravel(order="C").tobytes()
            )

    checkpoint_path = tmp_path / "model.npz"
    train(
        Namespace(
            data_dir=data_dir,
            grid_size=size,
            patch_size=3,
            layers=1,
            batch_size=1,
            samples_per_epoch=1,
            validation_samples=1,
            epochs=1,
            hidden_size=4,
            seed=0,
            learning_rate=1e-3,
            checkpoint=checkpoint_path,
        )
    )

    assert checkpoint_path.is_file()
    _, metadata = load_checkpoint(str(checkpoint_path))
    assert metadata["step"] == 1
