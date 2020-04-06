import pytest
import pytoolkit as tk


def test_CosineAnnealing():
    s = tk.schedules.CosineAnnealing(0.1, decay_steps=100, warmup_steps=10)
    assert s(0).numpy() == pytest.approx(0.01)
    assert s(4).numpy() == pytest.approx(0.05)
    assert s(9).numpy() == pytest.approx(0.1)
    assert s(50).numpy() == pytest.approx(0.05 * 1.01)
    assert s(100).numpy() == pytest.approx(0.001)
