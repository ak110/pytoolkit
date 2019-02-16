
import pytoolkit as tk


def test_get_custom_objects(dl_session):
    custom_objects = tk.get_custom_objects()
    assert custom_objects['NSGD'] == tk.optimizers.NSGD
    assert custom_objects['GroupNormalization'] == tk.layers.GroupNormalization
