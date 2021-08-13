import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


@pytest.mark.parametrize("mode", ["binary", "categorical"])
@pytest.mark.parametrize(
    "logit, phat_c",
    [(-10, 0.00), (0, 0.00), (+10, 0.00), (-10, 1.00), (0, 1.00), (+10, 1.00)],
)
def test_AutomatedFocalLoss(mode, logit, phat_c):
    if mode == "binary":
        input_shape = (None, 1)
        logits = np.array([[logit]], dtype=np.float32)
        y_true = np.array([[1]], dtype=np.float32)
    else:
        input_shape = (None, 2)
        logits = np.array([[logit, 0]], dtype=np.float32)
        y_true = np.array([[1, 0]], dtype=np.float32)

    layer = tk.layers.AutomatedFocalLoss(mode=mode)
    layer.build(input_shape=[input_shape, input_shape])
    tf.keras.backend.update(layer.phat_correct, np.array([phat_c]))
    loss = layer([y_true, logits]).numpy()
    if mode == "binary":
        assert loss.shape == (1, 1)
    else:
        assert loss.shape == (1,)
    assert loss > 0
    assert not np.isnan(loss).any()
    assert not np.isinf(loss).any()
    assert (layer.get_weights()[0] >= 0).all()
    assert (layer.get_weights()[0] <= 1).all()
