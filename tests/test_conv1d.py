import numpy as np

import heliax as hx


def test_conv1d_forward_and_gradcheck():
    generator = np.random.default_rng(1)
    layer = hx.Conv1d(2, 3, kernel_size=3, stride=2, padding=1, rng=generator)
    value = hx.tensor(generator.normal(size=(2, 2, 7)).astype(np.float32), requires_grad=True)
    output = layer(value)
    assert output.shape == (2, 3, 4)
    result = hx.gradcheck(lambda a: layer(a).sum(), [value], eps=0.01, atol=0.03, rtol=0.03)
    assert result["passed"], result
    output.sum().backward()
    assert layer.weight.grad is not None


def test_conv1d_dilation_and_padding():
    layer = hx.Conv1d(
        1, 1, kernel_size=2, padding=2, dilation=2, bias=False, rng=np.random.default_rng(2)
    )
    output = layer(hx.tensor(np.ones((1, 1, 5), dtype=np.float32)))
    assert output.shape == (1, 1, 7)
