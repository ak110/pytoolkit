import numpy as np
import pytest

import pytoolkit as tk


def test_top_k_accuracy():
    y_true = np.array([1, 1, 1])
    proba_pred = np.array([[0.2, 0.1, 0.3], [0.1, 0.2, 0.3], [0.1, 0.3, 0.2]])
    assert tk.ml.top_k_accuracy(y_true, proba_pred, k=2) == pytest.approx(2 / 3)
