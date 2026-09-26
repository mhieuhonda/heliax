import numpy as np

import heliax as hx


def test_attention_can_return_weights():
    layer = hx.MultiheadAttention(8, 2, rng=np.random.default_rng(1))
    value = hx.tensor(
        np.random.default_rng(2).normal(size=(2, 4, 8)).astype(np.float32), requires_grad=True
    )
    output, weights = layer(value, return_attention=True)
    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
    assert np.allclose(weights.numpy().sum(axis=-1), 1.0, atol=1e-5)
    output.sum().backward()
    assert value.grad is not None
