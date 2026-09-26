import numpy as np

import heliax as hx


def test_recursive_module_quantization():
    model = hx.nn.Sequential(
        hx.nn.Embedding(8, 4, rng=np.random.default_rng(1)),
        hx.nn.Linear(4, 2, rng=np.random.default_rng(2)),
    )
    hx.quantize_module(model, bits=8)
    assert isinstance(model.layers[0], hx.QuantizedEmbedding)
    assert isinstance(model.layers[1], hx.QuantizedLinear)
    output = model.layers[1](model.layers[0](hx.tensor([1, 2], dtype=np.int64)))
    assert output.shape == (2, 2)
