import numpy as np

import heliax as hx


def test_workspace_reuses_shape_dtype_buffers():
    with hx.workspace_context(max_buffers_per_key=2) as pool:
        first = pool.acquire((2, 3), dtype=np.float32)
        pool.release(first)
        second = pool.acquire((2, 3), dtype=np.float32)
        assert second is first
        report = pool.report()
        assert report["hits"] == 1
        assert report["misses"] == 1
        assert report["live_buffers"] == 1
    assert pool.report()["live_buffers"] == 0


def test_conv2d_uses_workspace_reuse():
    layer = hx.nn.Conv2d(1, 2, kernel_size=3, padding=1, rng=np.random.default_rng(1))
    value = hx.tensor(np.ones((1, 1, 4, 4), dtype=np.float32))
    layer(value)
    layer(value)
    assert layer._workspace.report()["hits"] >= 1
