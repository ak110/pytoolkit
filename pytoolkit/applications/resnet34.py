"""ResNet34。"""


def create(include_top=False, input_shape=None, input_tensor=None, weights="imagenet"):
    """ネットワークの作成。"""
    from classification_models.tfkeras import Classifiers

    backbone, _ = Classifiers.get("resnet34")
    return backbone(
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=weights,
    )


def preprocess_input(x):
    """前処理。"""
    return x


def get_1_over_2(model):
    """入力から縦横1/2のところのテンソルを返す。"""
    return model.get_layer("relu0").output


def get_1_over_4(model):
    """入力から縦横1/4のところのテンソルを返す。"""
    return model.get_layer("stage2_unit1_bn1").input


def get_1_over_8(model):
    """入力から縦横1/8のところのテンソルを返す。"""
    return model.get_layer("stage3_unit1_bn1").input


def get_1_over_16(model):
    """入力から縦横1/16のところのテンソルを返す。"""
    return model.get_layer("stage4_unit1_bn1").input


def get_1_over_32(model):
    """入力から縦横1/32のところのテンソルを返す。"""
    return model.output
