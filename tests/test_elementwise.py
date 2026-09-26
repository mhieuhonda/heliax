import numpy as np

import heliax as hx


def test_elementwise_abs_min_max_and_where():
    value = hx.tensor([-1.0, 2.0, -3.0], requires_grad=True)
    other = hx.tensor([0.5, 1.0, 4.0], requires_grad=True)
    result = hx.where(value > 0, value.minimum(other), value.abs())
    result.sum().backward()
    assert result.shape == value.shape
    assert value.grad is not None and other.grad is not None
    assert np.allclose(value.sign().numpy(), [-1.0, 1.0, -1.0])
