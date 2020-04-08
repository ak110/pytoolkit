"""se_resnext50_32x4d。HDF5ではsave出来ないので注意。"""

import tensorflow as tf


def create(include_top=False, input_shape=None, input_tensor=None, weights="imagenet"):
    """ネットワークの作成。"""
    if input_shape is None:
        assert input_tensor is not None
    else:
        assert input_tensor is None
        input_tensor = tf.keras.layers.Input(input_shape)
    assert not include_top
    assert weights in (None, "imagenet")

    from tf2cv.model_provider import get_model as tf2cv_get_model

    net = tf2cv_get_model(
        "seresnext50_32x4d", pretrained=weights is not None, data_format="channels_last"
    )
    x = net.features.get_layer("init_block")(input_tensor)
    x = net.features.get_layer("stage1")(x)
    x = net.features.get_layer("stage2")(x)
    x = net.features.get_layer("stage3")(x)
    x = net.features.get_layer("stage4")(x)

    inputs = tf.keras.utils.get_source_inputs(input_tensor)
    model = tf.keras.models.Model(inputs, x, name="darknet53")

    return model


def preprocess_input(x):
    """前処理。"""
    return tf.keras.applications.imagenet_utils.preprocess_input(x, mode="torch")


def get_1_over_4(model):
    """入力から縦横1/4のところのテンソルを返す。"""
    return model.get_layer("stage1").output


def get_1_over_8(model):
    """入力から縦横1/8のところのテンソルを返す。"""
    return model.get_layer("stage2").output


def get_1_over_16(model):
    """入力から縦横1/16のところのテンソルを返す。"""
    return model.get_layer("stage3").output


def get_1_over_32(model):
    """入力から縦横1/32のところのテンソルを返す。"""
    return model.get_layer("stage4").output
