import numpy as np
import pytest

import pytoolkit as tk

module = tk.applications.resnet34


def test_model():
    model = module.create(input_shape=(256, 256, 3), weights=None)
    assert tuple(module.get_1_over_2(model).shape[1:3]) == (128, 128)
    assert tuple(module.get_1_over_4(model).shape[1:3]) == (64, 64)
    assert tuple(module.get_1_over_8(model).shape[1:3]) == (32, 32)
    assert tuple(module.get_1_over_16(model).shape[1:3]) == (16, 16)
    assert tuple(module.get_1_over_32(model).shape[1:3]) == (8, 8)


def test_preprocess_input():
    from classification_models.tfkeras import Classifiers

    _, preprocess_input = Classifiers.get("resnet34")
    x = np.random.uniform(0, 1, size=(3, 32, 32, 3))
    assert module.preprocess_input(x) == pytest.approx(preprocess_input(x))
