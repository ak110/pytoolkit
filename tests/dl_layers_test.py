
import numpy as np
import pytest

import pytoolkit as tk


def test_destandarization_layer():
    with tk.dl.session():
        import keras
        Destandarization = tk.dl.layers.destandarization()
        inp = x = keras.layers.Input(shape=(1,))
        x = Destandarization(3, 5)(x)
        model = keras.models.Model(inputs=inp, outputs=x)
        pred = model.predict(np.array([[2]]))
        assert pred[0] == pytest.approx(2 * 5 + 3)


def test_weighted_mean_layer():
    with tk.dl.session():
        import keras
        WeightedMean = tk.dl.layers.weighted_mean()
        inp = x = [
            keras.layers.Input(shape=(1,)),
            keras.layers.Input(shape=(1,)),
            keras.layers.Input(shape=(1,))
        ]
        x = WeightedMean()(x)
        model = keras.models.Model(inputs=inp, outputs=x)
        pred = model.predict([np.array([1]), np.array([2]), np.array([3])])
        assert pred[0] == pytest.approx((1 + 2 + 3) / 3)
