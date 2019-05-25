
import numpy as np
import pytest

import pytoolkit as tk

K = tk.K


def test_binary_iou(session):
    y_true = K.constant(np.array([[0.0, 0.0, 1.0, 1.0]]))
    y_pred = K.constant(np.array([[0.0, 0.9, 0.8, 0.0]]))
    metric = session.run(tk.metrics.binary_iou(y_true, y_pred, threshold=0.8))
    assert metric == pytest.approx(1 / 3)
