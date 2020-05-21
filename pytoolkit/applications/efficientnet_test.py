import numpy as np
import pytest

import pytoolkit as tk

module = tk.applications.efficientnet


def test_model():
    model = module.create_b0(input_shape=(256, 256, 3), weights=None)
    assert tuple(module.get_1_over_2(model).shape[1:3]) == (128, 128)
    assert tuple(module.get_1_over_4(model).shape[1:3]) == (64, 64)
    assert tuple(module.get_1_over_8(model).shape[1:3]) == (32, 32)
    assert tuple(module.get_1_over_16(model).shape[1:3]) == (16, 16)
    assert tuple(module.get_1_over_32(model).shape[1:3]) == (8, 8)


def test_save_load(tmpdir):
    model = module.create_b0(input_shape=(256, 256, 3), weights=None)
    tk.models.save(model, str(tmpdir / "model.h5"))
    tk.models.load(str(tmpdir / "model.h5"))


def test_preprocess_input():
    import efficientnet.tfkeras as efn

    x = np.random.uniform(0, 1, size=(3, 32, 32, 3))
    assert module.preprocess_input(x) == pytest.approx(efn.preprocess_input(x))
