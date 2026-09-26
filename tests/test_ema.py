import numpy as np

import heliax as hx


def test_model_ema_tracks_and_restores_state():
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(1))
    ema = hx.ModelEMA(model, decay=0.5)
    original = model.weight.numpy().copy()
    with hx.no_grad():
        model.weight.data = model.weight.numpy() + 1.0
    ema.update(model)
    assert not np.allclose(ema.shadow["weight"], original)
    ema.copy_to(model)
    assert np.allclose(model.weight.numpy(), ema.shadow["weight"])
    state = ema.state_dict()
    restored = hx.ModelEMA(model, decay=0.9)
    restored.load_state_dict(state)
    assert restored.step_count == 1
