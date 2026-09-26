import numpy as np

import heliax as hx


def test_causal_mask_and_module_dtype_helpers():
    mask = hx.causal_mask(3)
    assert mask.shape == (3, 3)
    assert np.array_equal(mask.numpy(), np.triu(np.ones((3, 3), dtype=bool), 1))
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(1))
    assert model.float().parameters()[0].dtype == np.dtype(np.float32)
    assert model.half().parameters()[0].dtype == np.dtype(np.float16)
