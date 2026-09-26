import numpy as np

import heliax as hx


def test_optimizer_state_round_trip_restores_fresh_optimizer(tmp_path):
    model = hx.nn.Linear(2, 2, rng=np.random.default_rng(1))
    optimizer = hx.optim.AdamW(model.parameters(), lr=0.01)
    for parameter in model.parameters():
        parameter.grad = hx.Tensor(np.ones_like(parameter.numpy()))
    optimizer.step()
    path = tmp_path / "optimizer.npz"
    hx.save_checkpoint(path, model, optimizer)
    restored_model = hx.nn.Linear(2, 2, rng=np.random.default_rng(9))
    restored_optimizer = hx.optim.AdamW(restored_model.parameters(), lr=0.01)
    hx.load_checkpoint(path, restored_model, restored_optimizer)
    assert restored_optimizer.state
    assert any("first_moment" in values for values in restored_optimizer.state.values())
