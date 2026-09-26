import numpy as np

import heliax as hx


def test_pool1d_forward_and_gradients():
    generator = np.random.default_rng(1)
    value = hx.tensor(generator.normal(size=(2, 3, 6)).astype(np.float32), requires_grad=True)
    max_pool = hx.MaxPool1d(kernel_size=2, stride=2)
    avg_pool = hx.AvgPool1d(kernel_size=3, stride=3, padding=1)
    max_output = max_pool(value)
    avg_output = avg_pool(value)
    assert max_output.shape == (2, 3, 3)
    assert avg_output.shape == (2, 3, 2)
    (max_output.sum() + avg_output.sum()).backward()
    assert value.grad is not None
