import numpy as np

import heliax as hx


def test_groupnorm_forward_and_gradcheck():
    norm = hx.nn.GroupNorm(2, 4)
    value = hx.tensor(
        np.random.default_rng(6).normal(size=(3, 4, 5)).astype(np.float32), requires_grad=True
    )
    output = norm(value)
    assert output.shape == value.shape
    result = hx.gradcheck(lambda a: norm(a).sum(), [value], eps=0.01, atol=0.04, rtol=0.04)
    assert result["passed"], result
