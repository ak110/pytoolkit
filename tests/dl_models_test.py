import pathlib

import pytoolkit as tk


def test_params(tmpdir):
    with tk.dl.session():
        filepath = str(tmpdir.join('model.params.png'))

        import keras
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(32, input_shape=(16,)))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.BatchNormalization())
        tk.dl.models.plot_model_params(model, filepath)

        assert pathlib.Path(filepath).is_file()  # とりあえず存在チェックだけ

        assert tk.dl.models.count_trainable_params(model) == (16 * 32 + 32) + (32 * 32 + 32) + 32 * 2
