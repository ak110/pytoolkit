import numpy as np
import pytest

import pytoolkit as tk


def test_logit():
    x = [0.0, 0.5, 1.0]
    y = [-16.118095, 0, +16.118095]
    assert tk.math.logit(x) == pytest.approx(y, abs=1e-6)


def test_sigmoid_logit():
    x1 = np.arange(-3, 4)
    x2 = tk.math.sigmoid(x1)
    x3 = tk.math.logit(x2)
    assert x3 == pytest.approx(x1)
