import numpy as np

import heliax as hx


def test_checkpoint_rejects_unknown_format(tmp_path):
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(3))
    path = tmp_path / "bad.npz"
    hx.save_checkpoint(path, model)
    state = hx.load_state_dict(path)
    state["format_version"] = np.asarray(99, dtype=np.int64)
    hx.save_state_dict(path, state)
    try:
        hx.load_checkpoint(path, model)
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown checkpoint format was accepted")


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
