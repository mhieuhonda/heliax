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


def test_rich_checkpoint_metadata_round_trip(tmp_path):
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(4))
    metadata = {"hparams": {"lr": 0.01, "layers": [2, 4]}, "tags": ["a", "b"], "epoch": 2}
    path = tmp_path / "rich.npz"
    hx.save_checkpoint(path, model, metadata=metadata)
    restored = hx.load_checkpoint(path, model)
    assert restored["hparams"] == metadata["hparams"]
    assert restored["tags"] == metadata["tags"]
    assert restored["epoch"] == 2


def test_non_strict_state_dict_loading():
    model = hx.nn.Sequential(hx.nn.Linear(2, 2, rng=np.random.default_rng(1)), hx.nn.ReLU())
    state = {"layers.0.bias": np.ones(2, dtype=np.float32)}
    model.load_state_dict(state, strict=False)
    assert np.allclose(model.state_dict()["layers.0.bias"], 1.0)
