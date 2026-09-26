import numpy as np

import heliax as hx


def test_expand_repeat_roll_squeeze_gradients():
    value = hx.tensor([[1.0, 2.0]], requires_grad=True)
    expanded = value.expand(2, 2, 2)
    repeated = value.repeat(2, axis=0)
    rolled = value.roll(1, axis=1)
    squeezed = hx.tensor([[[1.0, 2.0]]], requires_grad=True).squeeze(0)
    unsqueezed = squeezed.unsqueeze(0)
    result = (
        expanded.sum()
        + repeated.sum()
        + rolled.sum()
        + squeezed.sum()
        + unsqueezed.sum()
        + value.repeat(2).sum()
    )
    result.backward()
    assert value.grad is not None
    assert squeezed.grad is not None
    assert np.isfinite(value.grad.numpy()).all()
