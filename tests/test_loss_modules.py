import numpy as np

import heliax as hx


def test_regression_and_binary_loss_modules():
    prediction = hx.tensor(
        np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32), requires_grad=True
    )
    target = hx.tensor(np.array([[0.0, -1.0], [1.0, 2.0]], dtype=np.float32))
    losses = [
        hx.nn.MSELoss()(prediction, target),
        hx.nn.L1Loss()(prediction, target),
        hx.nn.HuberLoss(delta=1.0)(prediction, target),
    ]
    for loss in losses:
        assert loss.shape == ()
    logits = hx.tensor(np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32), requires_grad=True)
    binary_target = hx.tensor(np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))
    bce = hx.nn.BCEWithLogitsLoss()(logits, binary_target)
    bce.backward()
    assert logits.grad is not None
    probabilities = hx.tensor(np.array([[0.2, 0.8]], dtype=np.float32))
    assert (
        hx.nn.BCELoss()(probabilities, hx.tensor(np.array([[0.0, 1.0]], dtype=np.float32))).shape
        == ()
    )
