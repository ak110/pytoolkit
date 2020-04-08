"""Darknet53。"""

import tensorflow as tf

from .. import hvd


def create(
    include_top=False,
    input_shape=None,
    input_tensor=None,
    weights="imagenet",
    for_small=False,
):
    """Darknet53。

    feature map例:
        - model.get_layer("block1_conv1_act").output  # 1/1
        - model.get_layer("block2_add").output  # 1/2
        - model.get_layer("block4_add").output  # 1/4
        - model.get_layer("block12_add").output  # 1/8
        - model.get_layer("block20_add").output  # 1/16
        - model.get_layer("block24_add").output  # 1/32

    """
    if input_shape is None:
        assert input_tensor is not None
    else:
        assert input_tensor is None
        input_tensor = tf.keras.layers.Input(input_shape)
    assert not include_top
    assert weights in (None, "imagenet")

    x = _darknet_body(input_tensor, for_small=for_small)
    inputs = tf.keras.utils.get_source_inputs(input_tensor)
    model = tf.keras.models.Model(inputs, x, name="darknet53")

    if weights == "imagenet":
        weights_path = hvd.get_file(
            "pytoolkit_darknet53_weights.h5",
            "https://github.com/ak110/object_detector/releases/download/v0.0.1/darknet53_weights.h5",
            file_hash="1a3857c961bcb77cd25ebbe0fcb346d4",
            cache_subdir="models",
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _darknet_body(x, for_small=False):
    """Darknent body having 52 Convolution2D layers"""
    x = _darknet_conv_bn_act(32, (3, 3), "block1_conv1")(x)
    x = _darknet_resblocks(x, 64, 1, block_index=2, downsampling=not for_small)
    x = _darknet_resblocks(x, 128, 2, block_index=3)
    x = _darknet_resblocks(x, 256, 8, block_index=5)
    x = _darknet_resblocks(x, 512, 8, block_index=13)
    x = _darknet_resblocks(x, 1024, 4, block_index=21)
    return x


def _darknet_resblocks(x, num_filters, num_blocks, block_index, downsampling=True):
    """Darknet53用Residual blocks"""
    if downsampling:
        x = tf.keras.layers.ZeroPadding2D(
            ((1, 0), (1, 0)), name=f"block{block_index}_pad"
        )(x)
        x = _darknet_conv_bn_act(
            num_filters, 3, strides=2, padding="valid", name=f"block{block_index}_conv"
        )(x)
    else:
        x = _darknet_conv_bn_act(num_filters, 3, name=f"block{block_index}_conv")(x)
    for i in range(num_blocks):
        sc = x
        x = _darknet_conv_bn_act(
            num_filters // 2, 1, name=f"block{block_index + i}_conv1"
        )(x)
        x = _darknet_conv_bn_act(
            num_filters // 1, 3, name=f"block{block_index + i}_conv2"
        )(x)
        x = tf.keras.layers.add([sc, x], name=f"block{block_index + i}_add")
    return x


def _darknet_conv_bn_act(filters, kernel_size, name, strides=1, padding="same"):
    """Darknet53用Conv2D+BN+Activation"""

    def _container(x):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),  # オリジナルは5e-4
            name=name,
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1, name=f"{name}_act")(x)
        return x

    return _container


def preprocess_input(x):
    """前処理。"""
    return x / 255


def get_1_over_1(model):
    """入力から縦横1/1のところのテンソルを返す。"""
    return model.get_layer("block1_conv1_act").output


def get_1_over_2(model):
    """入力から縦横1/2のところのテンソルを返す。"""
    return model.get_layer("block2_add").output


def get_1_over_4(model):
    """入力から縦横1/4のところのテンソルを返す。"""
    return model.get_layer("block4_add").output


def get_1_over_8(model):
    """入力から縦横1/8のところのテンソルを返す。"""
    return model.get_layer("block12_add").output


def get_1_over_16(model):
    """入力から縦横1/16のところのテンソルを返す。"""
    return model.get_layer("block20_add").output


def get_1_over_32(model):
    """入力から縦横1/32のところのテンソルを返す。"""
    return model.get_layer("block24_add").output
