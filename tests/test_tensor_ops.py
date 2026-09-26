import numpy as np

import heliax as hx


def test_tensor_minimum_maximum_and_clamp_gradients():
    value = hx.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    other = hx.tensor([0.0, 0.5, 1.0], requires_grad=True)
    loss = value.minimum(other).sum() + value.maximum(other).sum() + value.clamp(0.0, 1.0).sum()
    loss.backward()
    assert value.grad is not None
    assert other.grad is not None
    assert np.all(np.isfinite(value.grad.numpy()))
    with np.testing.assert_raises(ValueError):
        value.clamp(2.0, 1.0)
