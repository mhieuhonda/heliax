import numpy as np

import heliax as hx


def test_transformer_encoder_layer_and_positional_encoding():
    rng = np.random.default_rng(7)
    value = hx.tensor(rng.normal(size=(2, 5, 8)).astype(np.float32), requires_grad=True)
    positional = hx.PositionalEncoding(8, max_length=16, dropout=0.0)
    encoded = positional(value)
    layer = hx.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, rng=rng)
    output = layer(encoded)
    assert output.shape == value.shape
    output.square().mean().backward()
    assert value.grad is not None
    assert layer.self_attention.in_proj_weight.grad is not None
    assert "encoding" in positional.state_dict()


def test_transformer_encoder_stack():
    rng = np.random.default_rng(8)
    layer = hx.TransformerEncoderLayer(6, 2, dim_feedforward=12, dropout=0.0, rng=rng)
    encoder = hx.TransformerEncoder(layer, num_layers=2)
    value = hx.tensor(rng.normal(size=(2, 3, 6)).astype(np.float32), requires_grad=True)
    output = encoder(value)
    assert output.shape == value.shape
    output.sum().backward()
    assert value.grad is not None
