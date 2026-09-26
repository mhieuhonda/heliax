import numpy as np

import heliax as hx


def test_checkpoint_manager_keeps_best_and_recent(tmp_path):
    model = hx.nn.Linear(2, 1, rng=np.random.default_rng(1))
    optimizer = hx.optim.SGD(model.parameters(), lr=0.01)
    manager = hx.CheckpointManager(tmp_path, model, optimizer, keep_last=1, mode="min")
    manager.save(metric=2.0, metadata={"step": 1})
    manager.save(metric=1.0, metadata={"step": 2})
    manager.save(metric=1.5, metadata={"step": 3})
    assert manager.best_metric == 1.0
    assert manager.latest_path.exists()
    assert manager.best_path.exists()
    metadata = manager.load_best()
    assert metadata["step"] == 2
