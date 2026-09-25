import numpy as np

import heliax as hx


def test_gru_sequence_shapes_and_backward():
    gru = hx.GRU(4, 6, rng=np.random.default_rng(9))
    value = hx.tensor(
        np.random.default_rng(10).normal(size=(5, 3, 4)).astype(np.float32), requires_grad=True
    )
    output, hidden = gru(value)
    assert output.shape == (5, 3, 6)
    assert hidden.shape == (3, 6)
    output.square().mean().backward()
    assert value.grad is not None
    assert gru.weight_ih.grad is not None


def test_lstm_sequence_shapes_and_backward():
    lstm = hx.LSTM(4, 6, rng=np.random.default_rng(11))
    value = hx.tensor(
        np.random.default_rng(12).normal(size=(5, 3, 4)).astype(np.float32), requires_grad=True
    )
    output, (hidden, cell) = lstm(value)
    assert output.shape == (5, 3, 6)
    assert hidden.shape == cell.shape == (3, 6)
    output.sum().backward()
    assert value.grad is not None
