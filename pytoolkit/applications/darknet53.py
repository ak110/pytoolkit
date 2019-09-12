"""Darknet53。"""

from .. import keras, hvd


def darknet53(input_shape=None, input_tensor=None, weights="imagenet", for_small=False):
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
        input_tensor = keras.layers.Input(input_shape)
    assert weights in (None, "imagenet")

    x = darknet_body(input_tensor, for_small=for_small)
    inputs = keras.utils.get_source_inputs(input_tensor)
    model = keras.models.Model(inputs, x, name="darknet53")

    if weights == "imagenet":
        weights_path = hvd.get_file(
            "pytoolkit_darknet53_weights.h5",
            "https://github.com/ak110/object_detector/releases/download/v0.0.1/darknet53_weights.h5",
            file_hash="1a3857c961bcb77cd25ebbe0fcb346d4",
            cache_subdir="models",
        )
        model.load_weights(str(weights_path))
    elif weights is not None:
        model.load_weights(weights)

    return model


def darknet_body(x, for_small=False):
    """Darknent body having 52 Convolution2D layers"""
    x = darknet_conv_bn_act(32, (3, 3), "block1_conv1")(x)
    x = darknet_resblocks(x, 64, 1, block_index=2, downsampling=not for_small)
    x = darknet_resblocks(x, 128, 2, block_index=3)
    x = darknet_resblocks(x, 256, 8, block_index=5)
    x = darknet_resblocks(x, 512, 8, block_index=13)
    x = darknet_resblocks(x, 1024, 4, block_index=21)
    return x


def darknet_resblocks(x, num_filters, num_blocks, block_index, downsampling=True):
    """Darknet53用Residual blocks"""
    if downsampling:
        x = keras.layers.ZeroPadding2D(
            ((1, 0), (1, 0)), name=f"block{block_index}_pad"
        )(x)
        x = darknet_conv_bn_act(
            num_filters, 3, strides=2, padding="valid", name=f"block{block_index}_conv"
        )(x)
    else:
        x = darknet_conv_bn_act(num_filters, 3, name=f"block{block_index}_conv")(x)
    for i in range(num_blocks):
        sc = x
        x = darknet_conv_bn_act(
            num_filters // 2, 1, name=f"block{block_index + i}_conv1"
        )(x)
        x = darknet_conv_bn_act(
            num_filters // 1, 3, name=f"block{block_index + i}_conv2"
        )(x)
        x = keras.layers.add([sc, x], name=f"block{block_index + i}_add")
    return x


def darknet_conv_bn_act(filters, kernel_size, name, strides=1, padding="same"):
    """Darknet53用Conv2D+BN+Activation"""

    def _container(x):
        x = keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(1e-4),  # オリジナルは5e-4
            name=name,
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = keras.layers.LeakyReLU(alpha=0.1, name=f"{name}_act")(x)
        return x

    return _container


def preprocess_input(x):
    """前処理。"""
    return x / 255
