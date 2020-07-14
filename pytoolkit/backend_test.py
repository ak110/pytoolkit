import pytest
import tensorflow as tf

import pytoolkit as tk


def test_logit():
    x = tf.constant([0.0, 0.5, 1.0])
    y = [-16.118095, 0, +16.118095]
    logits = tk.backend.logit(x).numpy()
    assert logits == pytest.approx(y, abs=1e-6)


@pytest.mark.parametrize("graph", [False, True])
def test_reduce_losses(graph):
    func = tk.backend.reduce_losses
    if graph:
        func = tf.function(func)

    with pytest.raises(ValueError):
        tk.backend.reduce_losses([tf.ones((1, 1))])

    with pytest.raises(tf.errors.InvalidArgumentError):
        tk.backend.reduce_losses([tf.ones((1,)) / 0.0])

    x = tk.backend.reduce_losses([tf.ones((1,))])
    assert x.shape.rank == 0
    assert x.numpy() == pytest.approx(1.0)
