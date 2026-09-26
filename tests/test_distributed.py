import numpy as np

import heliax as hx


def test_distributed_sampler_partitions_indices():
    dataset = hx.TensorDataset(hx.tensor(np.arange(10, dtype=np.float32).reshape(10, 1)))
    rank0 = hx.DistributedSampler(len(dataset), num_replicas=3, rank=0, shuffle=False)
    rank1 = hx.DistributedSampler(len(dataset), num_replicas=3, rank=1, shuffle=False)
    assert list(rank0) == [0, 3, 6, 9]
    assert list(rank1) == [1, 4, 7]
    loader = hx.DataLoader(dataset, batch_size=2, sampler=rank0)
    assert len(loader) == 2
    assert next(iter(loader))[0].shape == (2, 1)


def test_gradient_reduction_simulation():
    parameter = hx.Parameter(np.ones(2, dtype=np.float32))
    parameter.grad = hx.tensor(np.array([2.0, 4.0], dtype=np.float32))
    hx.reduce_gradients([parameter], world_size=2)
    assert np.allclose(parameter.grad.numpy(), [1.0, 2.0])
