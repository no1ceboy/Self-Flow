import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.projectors import build_projector, count_parameters


PROJECTOR_KINDS = ("residual_mlp", "deep_mlp", "convnext")
PARAM_BUDGET = 3_000_000


@pytest.mark.parametrize("kind", PROJECTOR_KINDS)
@pytest.mark.parametrize("shape", [(4, 768), (4, 16, 768)])
def test_projector_forward_shapes_and_finiteness(kind, shape):
    x = jnp.ones(shape, dtype=jnp.float32)
    projector = build_projector(kind=kind, input_dim=shape[-1], hidden_dim=384)

    variables = projector.init(jax.random.PRNGKey(0), x, train=True)
    y = projector.apply(variables, x, train=True)

    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


@pytest.mark.parametrize("kind", PROJECTOR_KINDS)
def test_default_projectors_stay_under_3m_params(kind):
    x = jnp.ones((2, 256, 1152), dtype=jnp.float32)
    projector = build_projector(kind=kind, input_dim=1152, hidden_dim=384)

    variables = projector.init(jax.random.PRNGKey(0), x, train=True)
    assert count_parameters(variables["params"]) < PARAM_BUDGET


def test_projector_can_change_output_dim():
    x = jnp.ones((2, 8, 512), dtype=jnp.float32)
    projector = build_projector(
        kind="residual_mlp",
        input_dim=512,
        output_dim=768,
        hidden_dim=384,
    )

    variables = projector.init(jax.random.PRNGKey(0), x, train=True)
    y = projector.apply(variables, x, train=True)

    assert y.shape == (2, 8, 768)
    assert jnp.all(jnp.isfinite(y))
