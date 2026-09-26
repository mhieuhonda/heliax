import numpy as np

import heliax as hx


def test_dilated_conv_shape_and_gradcheck():
    layer = hx.nn.Conv2d(1, 2, kernel_size=3, dilation=2, rng=np.random.default_rng(8))
    value = hx.tensor(
        np.random.default_rng(9).normal(size=(1, 1, 7, 7)).astype(np.float32), requires_grad=True
    )
    output = layer(value)
    assert output.shape == (1, 2, 3, 3)
    result = hx.gradcheck(lambda a: layer(a).sum(), [value], eps=0.01, atol=0.05, rtol=0.05)
    assert result["passed"], result
