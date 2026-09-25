import numpy as np

import heliax as hx


def test_dataset_batching_and_fit():
    generator = np.random.default_rng(1)
    x = hx.tensor(generator.normal(size=(10, 2)).astype(np.float32))
    y = hx.tensor((x.numpy() * 0.5).astype(np.float32))
    dataset = hx.TensorDataset(x, y)
    loader = hx.DataLoader(dataset, batch_size=4, shuffle=True, generator=np.random.default_rng(2))
    assert len(loader) == 3
    batches = list(loader)
    assert batches[0][0].shape == (4, 2)
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(3))
    optimizer = hx.optim.SGD(model.parameters(), lr=0.03)
    history = hx.fit(model, loader, optimizer, hx.mean_squared_error, epochs=2)
    assert len(history) == 2
    assert all(np.isfinite(value) for value in history)


def test_state_dict_round_trip(tmp_path):
    model = hx.nn.MLP(2, [4], 1, rng=np.random.default_rng(4))
    path = hx.save_state_dict(tmp_path / "weights.npz", model.state_dict())
    restored = model.state_dict()
    loaded = hx.load_state_dict(path)
    assert set(restored) == set(loaded)
    assert np.allclose(restored["network.layers.0.weight"], loaded["network.layers.0.weight"])
    hx.save_checkpoint(tmp_path / "checkpoint.npz", model, metadata={"epoch": 3})
    metadata = hx.load_checkpoint(tmp_path / "checkpoint.npz", model)
    assert metadata["epoch"] == 3


def test_profiler():
    results = []
    with hx.profile("work", results):
        _ = hx.tensor([1.0, 2.0]).sum()
    assert len(results) == 1
    assert results[0].name == "work"
    assert results[0].elapsed_ms >= 0
