"""Xception。"""
import functools

import tensorflow as tf

from .. import hvd, ndimage

K = tf.keras.backend


def create(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    for_small=False,
):
    """Xception。

    padding="same"にしたりinitializer, regularizerを指定したりしたもの。

    feature map例:
        - model.get_layer("block1_conv2_act").output  # 1/2
        - model.get_layer("block3_sepconv1_act").input  # 1/4
        - model.get_layer("block4_sepconv1_act").input  # 1/8
        - model.get_layer("block13_sepconv1_act").input  # 1/16
        - model.get_layer("block14_sepconv2_act").output  # 1/32

    """
    assert K.image_data_format() == "channels_last"
    assert not include_top

    if input_shape is None:
        assert input_tensor is not None
    else:
        assert input_tensor is None
        input_tensor = tf.keras.layers.Input(input_shape)

    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
    )
    sepconv2d = functools.partial(
        tf.keras.layers.SeparableConv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
    )
    bn = functools.partial(tf.keras.layers.BatchNormalization)
    act = functools.partial(tf.keras.layers.Activation, "relu")
    mp = functools.partial(
        tf.keras.layers.MaxPooling2D, pool_size=3, strides=2, padding="same"
    )
    ap = functools.partial(
        tf.keras.layers.AveragePooling2D, pool_size=3, strides=2, padding="same"
    )

    first_strides = 1 if for_small else 2
    x = conv2d(32, strides=first_strides, name="block1_conv1")(input_tensor)
    x = bn(name="block1_conv1_bn")(x)
    x = act(name="block1_conv1_act")(x)
    x = conv2d(64, name="block1_conv2")(x)
    x = bn(name="block1_conv2_bn")(x)
    x = act(name="block1_conv2_act")(x)

    residual = ap(pool_size=2, strides=1)(x)  # 怪しいアレンジ
    residual = conv2d(128, kernel_size=1, strides=2, name="block2_conv")(residual)
    residual = bn(name="block2_conv_bn")(residual)

    x = sepconv2d(128, name="block2_sepconv1")(x)
    x = bn(name="block2_sepconv1_bn")(x)
    x = act(name="block2_sepconv2_act")(x)
    x = sepconv2d(128, name="block2_sepconv2")(x)
    x = bn(name="block2_sepconv2_bn")(x)

    x = mp(name="block2_pool")(x)
    x = tf.keras.layers.add([x, residual], name="block2_add")

    residual = ap(pool_size=2, strides=1)(x)  # 怪しいアレンジ
    residual = conv2d(256, kernel_size=1, strides=2, name="block3_conv")(residual)
    residual = bn(name="block3_conv_bn")(residual)

    x = act(name="block3_sepconv1_act")(x)
    x = sepconv2d(256, name="block3_sepconv1")(x)
    x = bn(name="block3_sepconv1_bn")(x)
    x = act(name="block3_sepconv2_act")(x)
    x = sepconv2d(256, name="block3_sepconv2")(x)
    x = bn(name="block3_sepconv2_bn")(x)

    x = mp(name="block3_pool")(x)
    x = tf.keras.layers.add([x, residual], name="block3_add")

    residual = ap(pool_size=2, strides=1)(x)  # 怪しいアレンジ
    residual = conv2d(728, kernel_size=1, strides=2, name="block4_conv")(residual)
    residual = bn(name="block4_conv_bn")(residual)

    x = act(name="block4_sepconv1_act")(x)
    x = sepconv2d(728, name="block4_sepconv1")(x)
    x = bn(name="block4_sepconv1_bn")(x)
    x = act(name="block4_sepconv2_act")(x)
    x = sepconv2d(728, name="block4_sepconv2")(x)
    x = bn(name="block4_sepconv2_bn")(x)

    x = mp(name="block4_pool")(x)
    x = tf.keras.layers.add([x, residual], name="block4_add")

    for i in range(8):
        residual = x
        prefix = "block" + str(i + 5)

        x = act(name=prefix + "_sepconv1_act")(x)
        x = sepconv2d(728, name=prefix + "_sepconv1")(x)
        x = bn(name=prefix + "_sepconv1_bn")(x)
        x = act(name=prefix + "_sepconv2_act")(x)
        x = sepconv2d(728, name=prefix + "_sepconv2")(x)
        x = bn(name=prefix + "_sepconv2_bn")(x)
        x = act(name=prefix + "_sepconv3_act")(x)
        x = sepconv2d(728, name=prefix + "_sepconv3")(x)
        x = bn(name=prefix + "_sepconv3_bn")(x)

        x = tf.keras.layers.add([x, residual], name=prefix + "_add")

    residual = ap(pool_size=2, strides=1)(x)  # 怪しいアレンジ
    residual = conv2d(1024, kernel_size=1, strides=2, name="block13_conv")(residual)
    residual = bn(name="block13_conv_bn")(residual)

    x = act(name="block13_sepconv1_act")(x)
    x = sepconv2d(728, name="block13_sepconv1")(x)
    x = bn(name="block13_sepconv1_bn")(x)
    x = act(name="block13_sepconv2_act")(x)
    x = sepconv2d(1024, name="block13_sepconv2")(x)
    x = bn(name="block13_sepconv2_bn")(x)

    x = mp(name="block13_pool")(x)
    x = tf.keras.layers.add([x, residual], name="block13_add")

    x = sepconv2d(1536, name="block14_sepconv1")(x)
    x = bn(name="block14_sepconv1_bn")(x)
    x = act(name="block14_sepconv1_act")(x)

    x = sepconv2d(2048, name="block14_sepconv2")(x)
    x = bn(name="block14_sepconv2_bn")(x)
    x = act(name="block14_sepconv2_act")(x)

    inputs = tf.keras.utils.get_source_inputs(input_tensor)
    model = tf.keras.models.Model(inputs, x, name="xception")

    if weights == "imagenet":
        weights_path = hvd.get_file(
            "xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
            "https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
            cache_subdir="models",
            file_hash="b0042744bf5b25fce3cb969f33bebb97",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x):
    """前処理。"""
    return ndimage.preprocess_tf(x)


def get_1_over_2(model):
    """入力から縦横1/2のところのテンソルを返す。"""
    return model.get_layer("block1_conv2_act").output


def get_1_over_4(model):
    """入力から縦横1/4のところのテンソルを返す。"""
    return model.get_layer("block3_sepconv1_act").input


def get_1_over_8(model):
    """入力から縦横1/8のところのテンソルを返す。"""
    return model.get_layer("block4_sepconv1_act").input


def get_1_over_16(model):
    """入力から縦横1/16のところのテンソルを返す。"""
    return model.get_layer("block13_sepconv1_act").input


def get_1_over_32(model):
    """入力から縦横1/32のところのテンソルを返す。"""
    return model.output
