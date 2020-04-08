"""EfficientNet。

参考: 本来の入力サイズ

- B0: 224
- B1: 240
- B2: 260
- B3: 300
- B4: 380
- B5: 456
- B6: 528
- B7: 600


"""

import tensorflow as tf


def create_b0(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB0(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b1(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB1(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b2(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB2(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b3(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB3(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b4(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB4(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b5(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB5(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b6(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB6(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def create_b7(
    include_top=False, input_shape=None, input_tensor=None, weights="noisy-student"
):
    """ネットワークの作成。"""
    import efficientnet.tfkeras as efn

    return efn.EfficientNetB7(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def preprocess_input(x):
    """前処理。"""
    return tf.keras.applications.imagenet_utils.preprocess_input(x, mode="torch")


def get_1_over_2(model):
    """入力から縦横1/2のところのテンソルを返す。"""
    return model.get_layer("block2a_expand_conv").input


def get_1_over_4(model):
    """入力から縦横1/4のところのテンソルを返す。"""
    return model.get_layer("block3a_expand_conv").input


def get_1_over_8(model):
    """入力から縦横1/8のところのテンソルを返す。"""
    return model.get_layer("block4a_expand_conv").input


def get_1_over_16(model):
    """入力から縦横1/16のところのテンソルを返す。"""
    return model.get_layer("block5a_expand_conv").input


def get_1_over_32(model):
    """入力から縦横1/32のところのテンソルを返す。"""
    return model.get_layer("top_activation").output
