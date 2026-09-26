import numpy as np

import heliax as hx


def test_grouped_conv_forward_and_backward():
    layer = hx.GroupedConv2d(4, 6, groups=2, kernel_size=3, padding=1, rng=np.random.default_rng(6))
    value = hx.tensor(
        np.random.default_rng(7).normal(size=(2, 4, 5, 5)).astype(np.float32), requires_grad=True
    )
    output = layer(value)
    assert output.shape == (2, 6, 5, 5)
    output.sum().backward()
    assert value.grad is not None
    assert len(layer.parameters()) == 4

    dilated = hx.GroupedConv2d(
        4, 4, groups=2, kernel_size=3, padding=1, dilation=2, rng=np.random.default_rng(8)
    )
    assert dilated(value.detach()).shape == (2, 4, 3, 3)
