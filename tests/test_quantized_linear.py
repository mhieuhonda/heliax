import numpy as np

import heliax as hx


def test_quantized_linear_inference_and_state():
    layer = hx.QuantizedLinear(4, 3, bits=8, rng=np.random.default_rng(1))
    value = hx.tensor(np.random.default_rng(2).normal(size=(5, 4)).astype(np.float32))
    output = layer(value)
    assert output.shape == (5, 3)
    assert "weight_int8" in layer.state_dict()
    assert layer.dequantized_weight().shape == (3, 4)
