import numpy as np

import heliax as hx


def test_fan_aware_initializers():
    value = hx.Parameter(np.zeros((8, 4), dtype=np.float32))
    hx.kaiming_uniform_(value, rng=np.random.default_rng(1))
    assert np.abs(value.numpy()).max() > 0
    hx.xavier_uniform_(value, rng=np.random.default_rng(2))
    hx.normal_(value, std=0.1, rng=np.random.default_rng(3))
    hx.uniform_(value, 0.01, rng=np.random.default_rng(4))
    hx.zeros_(value)
    assert np.all(value.numpy() == 0)
    hx.ones_(value)
    assert np.all(value.numpy() == 1)
