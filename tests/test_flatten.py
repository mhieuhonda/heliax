import numpy as np

import heliax as hx


def test_flatten_supports_arbitrary_dimensions():
    value = hx.tensor(np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5))
    assert hx.nn.Flatten(1)(value).shape == (2, 60)
    assert hx.nn.Flatten(2, 3)(value).shape == (2, 3, 20)
