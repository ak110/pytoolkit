
import pytoolkit as tk


def test_get_custom_objects():
    custom_objects = tk.dl.get_custom_objects()
    assert str(custom_objects['Destandarization']) == str(tk.dl.layers.destandarization())
    assert str(custom_objects['StocasticAdd']) == str(tk.dl.layers.stocastic_add())
    assert str(custom_objects['L2Normalization']) == str(tk.dl.layers.l2normalization())
    assert str(custom_objects['WeightedMean']) == str(tk.dl.layers.weighted_mean())
