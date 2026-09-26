import numpy as np

import heliax as hx


def test_unbind_and_chunk_shapes_and_gradients():
    value = hx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)
    parts = value.unbind(axis=0)
    assert [part.shape for part in parts] == [(4,), (4,), (4,)]
    sum(part.sum() for part in parts).backward()
    assert value.grad is not None
    value.zero_grad()
    chunks = value.chunk(2, axis=1)
    assert [chunk.shape for chunk in chunks] == [(3, 2), (3, 2)]
    sum(chunk.mean() for chunk in chunks).backward()
    assert value.grad is not None
