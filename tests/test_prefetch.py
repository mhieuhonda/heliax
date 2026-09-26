import numpy as np

import heliax as hx


def test_prefetch_loader_preserves_batches():
    features = hx.tensor(np.arange(10, dtype=np.float32).reshape(10, 1))
    loader = hx.DataLoader(hx.TensorDataset(features), batch_size=3)
    prefetched = list(hx.PrefetchLoader(loader, depth=2))
    assert len(prefetched) == len(loader) == 4
    assert np.array_equal(
        np.concatenate([batch[0].numpy() for batch in prefetched]), features.numpy()
    )
