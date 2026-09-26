import numpy as np

import heliax as hx


def test_quantized_embedding_inference_and_bounds():
    layer = hx.QuantizedEmbedding(8, 4, bits=8, rng=np.random.default_rng(1))
    output = layer(hx.tensor([1, 3, 5], dtype=np.int64))
    assert output.shape == (3, 4)
    assert "weight_int8" in layer.state_dict()
    with np.testing.assert_raises(IndexError):
        layer(np.array([8], dtype=np.int64))
