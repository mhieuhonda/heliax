import numpy as np

import heliax as hx


def test_cosine_warm_restarts_schedule():
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(1))
    optimizer = hx.optim.SGD(model.parameters(), lr=0.1)
    scheduler = hx.CosineAnnealingWarmRestarts(
        optimizer, first_cycle_steps=4, min_lr=0.01, factor=0.5
    )
    values = [scheduler.step() for _ in range(9)]
    assert values[0] > values[2] > values[3]
    assert values[3] < 0.03
    assert values[4] > values[5]
    assert np.isclose(optimizer.defaults["lr"], values[-1])
