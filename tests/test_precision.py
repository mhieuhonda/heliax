import numpy as np

import heliax as hx


def test_grad_scaler_scales_and_updates_optimizer():
    parameter = hx.Parameter(np.ones(2, dtype=np.float32))
    optimizer = hx.optim.SGD([parameter], lr=0.1)
    scaler = hx.GradScaler(initial_scale=10.0, growth_interval=2)
    loss = (parameter * 2).sum()
    scaled = scaler.scale_loss(loss)
    scaled.backward()
    scaler.unscale_gradients([parameter])
    assert np.allclose(parameter.grad.numpy(), 2.0)
    scaler.step(optimizer, [parameter])
    assert np.all(parameter.numpy() < 1.0)
    state = scaler.state_dict()
    restored = hx.GradScaler()
    restored.load_state_dict(state)
    assert restored.scale == 10.0


def test_autocast_is_explicit():
    with hx.autocast(np.float16) as dtype:
        assert dtype == np.float16
    with hx.autocast(np.float16, enabled=False) as dtype:
        assert dtype is None
