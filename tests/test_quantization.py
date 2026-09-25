import numpy as np

import heliax as hx


def test_symmetric_and_affine_quantization_round_trip():
    value = hx.tensor(np.array([[-1.0, -0.2], [0.1, 1.0]], dtype=np.float32))
    symmetric = hx.quantize(value, bits=8, symmetric=True)
    restored = hx.dequantize(symmetric)
    assert symmetric.shape == value.shape
    assert np.max(np.abs(restored.numpy() - value.numpy())) < 0.01

    affine = hx.quantize(value, bits=8, symmetric=False)
    restored = hx.dequantize(affine)
    assert np.max(np.abs(restored.numpy() - value.numpy())) < 0.01


def test_quantized_state_dict_and_compression():
    model = hx.nn.Linear(8, 8, rng=np.random.default_rng(4))
    state = hx.quantize_state_dict(model, bits=8)
    restored = hx.dequantize_state_dict(state)
    assert set(restored) == set(model.state_dict())
    assert hx.compression_ratio(model) > 1.0
