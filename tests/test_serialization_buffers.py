import numpy as np

import heliax as hx


def test_buffer_round_trip_in_checkpoint(tmp_path):
    model = hx.nn.Sequential(
        hx.nn.BatchNorm1d(3),
        hx.nn.Linear(3, 2, rng=np.random.default_rng(1)),
    )
    value = hx.tensor(np.random.default_rng(2).normal(size=(8, 3)).astype(np.float32))
    model(value)
    path = tmp_path / "buffered.npz"
    hx.save_checkpoint(path, model, metadata={"kind": "buffer-test"})
    restored = hx.nn.Sequential(
        hx.nn.BatchNorm1d(3),
        hx.nn.Linear(3, 2, rng=np.random.default_rng(9)),
    )
    metadata = hx.load_checkpoint(path, restored)
    assert metadata["kind"] == "buffer-test"
    for key, expected in model.state_dict().items():
        assert np.allclose(restored.state_dict()[key], expected), key
