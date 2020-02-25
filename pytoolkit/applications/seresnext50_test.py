import pytoolkit as tk


def test_model():
    module = tk.applications.seresnext50
    model = module.create(input_shape=(256, 256, 3), weights=None)
    assert tuple(module.get_1_over_4(model).shape[1:3]) == (64, 64)
    assert tuple(module.get_1_over_8(model).shape[1:3]) == (32, 32)
    assert tuple(module.get_1_over_16(model).shape[1:3]) == (16, 16)
    assert tuple(module.get_1_over_32(model).shape[1:3]) == (8, 8)
