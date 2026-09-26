import numpy as np

import heliax as hx


def test_reduce_lr_on_plateau():
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(1))
    optimizer = hx.optim.SGD(model.parameters(), lr=0.1)
    scheduler = hx.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    assert scheduler.step(1.0) == 0.1
    assert scheduler.step(1.5) == 0.1
    assert scheduler.step(1.6) == 0.05
    assert np.isclose(optimizer.defaults["lr"], 0.05)
