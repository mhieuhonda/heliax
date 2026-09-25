import numpy as np

import heliax as hx


def test_max_and_avg_pool_forward_shapes():
    value = hx.tensor(
        np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4), requires_grad=True
    )
    maximum = hx.nn.MaxPool2d(2)(value)
    average = hx.nn.AvgPool2d(2)(value)
    assert maximum.shape == (1, 1, 2, 2)
    assert average.shape == (1, 1, 2, 2)
    assert maximum.numpy()[0, 0, 0, 0] == 5
    assert np.allclose(average.numpy(), [[[2.5, 4.5], [10.5, 12.5]]])


def test_pooling_gradients():
    value = hx.tensor(
        np.random.default_rng(4).normal(size=(1, 1, 4, 4)).astype(np.float32), requires_grad=True
    )
    result = hx.gradcheck(
        lambda a: hx.nn.MaxPool2d(2)(a).sum(), [value], eps=0.01, atol=0.04, rtol=0.04
    )
    assert result["passed"], result
    value.zero_grad()
    hx.nn.AvgPool2d(2)(value).sum().backward()
    assert value.grad.shape == value.shape
