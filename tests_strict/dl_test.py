import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_destandarization_layer():
    import keras

    Destandarization = tk.dl.layers.destandarization()
    inp = x = keras.layers.Input(shape=(1,))
    x = Destandarization(3, 5)(x)
    model = keras.models.Model(inputs=inp, outputs=x)
    pred = model.predict(np.array([[2]]))
    assert pred[0] == pytest.approx(2 * 5 + 3)


def test_weighted_mean_layer():
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


def test_xor(tmpdir):
    import keras

    logger = tk.log.get('test_xor')
    logger.addHandler(tk.log.file_handler(str(tmpdir.join('test_xor.log'))))

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    with tk.dl.session():
        builder = tk.dl.layers.Builder()
        inp = x = keras.layers.Input(shape=(2,))
        x = builder.dense(16, use_bias=False)(x)
        x = builder.bn_act()(x)
        x = builder.dense(1, activation='sigmoid')(x)
        model = keras.models.Model(inputs=inp, outputs=x)
        model.compile('adam', 'binary_crossentropy', ['acc'])
        model.fit(
            X.repeat(4096, axis=0),
            y.repeat(4096, axis=0),
            epochs=8,
            verbose=2,
            callbacks=[
                tk.dl.callbacks.learning_rate(logger_name='test_xor'),
                tk.dl.callbacks.learning_curve_plot(str(tmpdir.join('history.png'))),
                tk.dl.callbacks.tsv_logger(str(tmpdir.join('history.tsv'))),
                tk.dl.callbacks.epoch_logger('test_xor'),
                tk.dl.callbacks.freeze_bn(0.5, logger_name='test_xor'),
            ])
        pred = model.predict(X)
        y_pred = (pred > 0.5).astype(np.int32).reshape(y.shape)
    assert (y_pred == y).all()


def test_get_custom_objects():
    custom_objects = tk.dl.get_custom_objects()
    assert str(custom_objects['Destandarization']) == str(tk.dl.layers.destandarization())
    assert str(custom_objects['StocasticAdd']) == str(tk.dl.layers.stocastic_add())
    assert str(custom_objects['L2Normalization']) == str(tk.dl.layers.l2normalization())
    assert str(custom_objects['WeightedMean']) == str(tk.dl.layers.weighted_mean())


def test_params(tmpdir):
    filepath = str(tmpdir.join('model.params.png'))

    import keras
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(16,)))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.BatchNormalization())
    tk.dl.models.plot_model_params(model, filepath)

    assert pathlib.Path(filepath).is_file()  # とりあえず存在チェックだけ

    assert tk.dl.models.count_trainable_params(model) == (16 * 32 + 32) + (32 * 32 + 32) + 32 * 2
