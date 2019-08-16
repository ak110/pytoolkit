import pytoolkit as tk


def test_get_custom_objects():
    d = tk.get_custom_objects()
    assert d["GroupNormalization"] == tk.layers.GroupNormalization
