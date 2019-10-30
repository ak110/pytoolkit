import numpy as np
import pytest
import sklearn.metrics.pairwise

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


def test_cosine_most_similars():
    v1 = np.random.normal(size=(456, 3))
    v2 = np.random.normal(size=(123, 3))

    indices, similarities = tk.math.cosine_most_similars(v1, v2, batch_size=9)

    cs = sklearn.metrics.pairwise.cosine_similarity(
        v1.astype(np.float32), v2.astype(np.float32)
    )
    indices_true = cs.argmax(axis=-1)
    similarities_true = cs.max(axis=-1)

    assert indices == pytest.approx(indices_true)
    assert similarities == pytest.approx(similarities_true)
