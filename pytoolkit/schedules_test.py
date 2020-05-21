import pytest

import pytoolkit as tk


def test_CosineAnnealing():
    s = tk.schedules.CosineAnnealing(0.1, decay_steps=100, warmup_steps=10)
    assert s(0).numpy() == pytest.approx(0.01)
    assert s(4).numpy() == pytest.approx(0.05)
    assert s(9).numpy() == pytest.approx(0.1)
    assert s(54).numpy() == pytest.approx(0.05 * 1.01)
    assert s(99).numpy() == pytest.approx(0.001)
    assert s(100).numpy() == pytest.approx(0.001)


def test_LinearDecay():
    s = tk.schedules.LinearDecay(0.1, decay_steps=100, warmup_steps=10)
    assert s(0).numpy() == pytest.approx(0.01)
    assert s(4).numpy() == pytest.approx(0.05)
    assert s(9).numpy() == pytest.approx(0.1)
    assert s(39).numpy() == pytest.approx(0.1 * (2 / 3) + 0.001 * (1 / 3))
    assert s(54).numpy() == pytest.approx(0.1 * (1 / 2) + 0.001 * (1 / 2))
    assert s(69).numpy() == pytest.approx(0.1 * (1 / 3) + 0.001 * (2 / 3))
    assert s(99).numpy() == pytest.approx(0.001)
    assert s(100).numpy() == pytest.approx(0.001)


def test_ExponentialDecay():
    s = tk.schedules.ExponentialDecay(0.1, decay_steps=100, warmup_steps=10)
    assert s(0).numpy() == pytest.approx(0.01)
    assert s(4).numpy() == pytest.approx(0.05)
    assert s(9).numpy() == pytest.approx(0.1)
    assert s(10).numpy() == pytest.approx(0.1 * 0.01 ** (1 / 90))
    assert s(11).numpy() == pytest.approx(0.1 * 0.01 ** (2 / 90))
    assert s(12).numpy() == pytest.approx(0.1 * 0.01 ** (3 / 90))
    assert s(99).numpy() == pytest.approx(0.001)
    assert s(100).numpy() == pytest.approx(0.001)
