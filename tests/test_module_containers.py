import numpy as np

import heliax as hx


def test_module_list_and_dict_state_traversal():
    layers = hx.nn.ModuleList(hx.nn.Linear(2, 3, rng=np.random.default_rng(1)), hx.nn.ReLU())
    value = hx.tensor(np.ones((4, 2), dtype=np.float32))
    output = layers[1](layers[0](value))
    assert output.shape == (4, 3)
    assert len(list(layers.parameters())) == 2
    registry = hx.nn.ModuleDict({"first": hx.nn.Linear(2, 2, rng=np.random.default_rng(2))})
    assert registry["first"](value).shape == (4, 2)
    assert "items.first.weight" in registry.state_dict()
