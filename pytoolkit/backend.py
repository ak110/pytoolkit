"""keras.backendやtensorflowの基礎的な関数など。"""

from . import K


def logit(x):
    """ロジット関数。シグモイド関数の逆関数。"""
    x = K.clip(x, K.epsilon(), 1 - K.epsilon())
    return K.log(x / (1 - x))
