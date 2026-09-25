import numpy as np

import heliax as hx


def test_masked_softmax_excludes_masked_entries():
    value = hx.tensor(
        np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 2.0]], dtype=np.float32), requires_grad=True
    )
    mask = np.array([[False, True, False], [True, True, True]])
    output = hx.masked_softmax(value, mask, axis=-1)
    assert np.allclose(output.numpy()[0, 1], 0.0)
    assert np.allclose(output.numpy()[0, 0] + output.numpy()[0, 2], 1.0)
    output.sum().backward()
    assert value.grad is not None


def test_scaled_dot_product_attention_shapes_and_gradients():
    rng = np.random.default_rng(5)
    query = hx.tensor(rng.normal(size=(2, 3, 4)).astype(np.float32), requires_grad=True)
    key = hx.tensor(rng.normal(size=(2, 5, 4)).astype(np.float32), requires_grad=True)
    value = hx.tensor(rng.normal(size=(2, 5, 6)).astype(np.float32), requires_grad=True)
    output = hx.scaled_dot_product_attention(query, key, value)
    assert output.shape == (2, 3, 6)
    output.square().mean().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
