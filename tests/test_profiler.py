import numpy as np

import heliax as hx


def test_graph_summary_and_memory_report():
    value = hx.tensor(np.ones((3, 4), dtype=np.float32), requires_grad=True)
    output = (value * 2.0).relu().sum()
    summary = hx.graph_summary(output)
    assert summary["nodes"] >= 3
    assert summary["storage_bytes"] > 0
    assert hx.op_histogram(output)["mul"] == 1
    model = hx.nn.Linear(4, 2, rng=np.random.default_rng(1))
    report = hx.memory_report(model)
    assert report["parameters"] == 10
    assert report["parameter_bytes"] > 0
    assert hx.memory_bytes(model.weight) == 4 * 2 * np.dtype(np.float32).itemsize
    summary = hx.model_summary(model)
    assert {row["name"] for row in summary} == {"weight", "bias"}


def test_gradient_and_training_memory_reports():
    model = hx.nn.Linear(4, 2, rng=np.random.default_rng(2))
    values = hx.tensor(np.ones((3, 4), dtype=np.float32))
    loss = model(values).sum()
    before = hx.gradient_memory_report(model)
    assert before["gradients"] == 0
    assert before["missing_gradients"] == 2
    loss.backward()
    after = hx.gradient_memory_report(model)
    assert after["gradients"] == 2
    assert after["gradient_bytes"] > 0
    training = hx.training_memory_report(model, loss)
    assert training["graph_nodes"] > 0
    assert training["total_training_bytes"] > training["parameter_bytes"]
