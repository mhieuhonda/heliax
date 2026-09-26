import numpy as np

import heliax as hx


def test_trainer_fit_evaluate_and_state_round_trip(tmp_path):
    generator = np.random.default_rng(1)
    features = hx.tensor(generator.normal(size=(16, 3)).astype(np.float32))
    targets = hx.tensor((features.numpy() * 0.5).astype(np.float32))
    loader = hx.DataLoader(hx.TensorDataset(features, targets), batch_size=4, shuffle=False)
    model = hx.nn.Linear(3, 1, rng=generator)
    optimizer = hx.optim.SGD(model.parameters(), lr=0.03)
    trainer = hx.Trainer(
        model, optimizer, hx.mean_squared_error, epochs=2, gradient_accumulation_steps=2
    )
    history = trainer.fit(loader)
    assert len(history) == 2
    assert all("loss" in row for row in history)
    assert trainer.evaluate(loader)["loss"] >= 0
    state = trainer.state_dict()
    assert state["model"]
    trainer.save(str(tmp_path / "trainer.npz"))
    trainer.load(str(tmp_path / "trainer.npz"))
    assert trainer.global_step == state["global_step"]
    scaler = hx.GradScaler(initial_scale=2.0)
    scaled_trainer = hx.Trainer(
        model, optimizer, hx.mean_squared_error, epochs=1, grad_scaler=scaler
    )
    scaled_trainer.fit(loader)
    assert scaler.scale == 2.0
