import jax
import jax.numpy as jnp
import numpy as np

from cfd_gnn.model import (
    apply_model,
    initialize_parameters,
    load_checkpoint,
    save_checkpoint,
)


def test_message_passing_model_preserves_grid_shape_and_is_differentiable():
    params = initialize_parameters(jax.random.PRNGKey(0), 5, 8, 2, 2)
    features = jnp.ones((1, 6, 6, 6, 5), dtype=jnp.float32)

    output = apply_model(params, features)
    gradient = jax.grad(lambda x: jnp.sum(apply_model(params, x)))(features)

    assert output.shape == (1, 6, 6, 6, 2)
    assert gradient.shape == features.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_checkpoint_round_trip(tmp_path):
    params = initialize_parameters(jax.random.PRNGKey(1), 4, 6, 1, 1)
    path = tmp_path / "checkpoint.npz"
    mean = jnp.zeros((1,))
    standard_deviation = jnp.ones((1,))

    save_checkpoint(
        str(path), params, mean, standard_deviation, 1.0, 4, 6, 1, 3
    )
    restored, metadata = load_checkpoint(str(path))

    assert metadata["step"] == 3
    np.testing.assert_array_equal(
        apply_model(restored, jnp.ones((1, 4, 4, 4, 4))),
        apply_model(params, jnp.ones((1, 4, 4, 4, 4))),
    )
