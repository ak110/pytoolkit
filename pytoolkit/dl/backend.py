"""keras.backendやtensorflowの基礎的な関数など。"""

import numpy as np


def logit(x):
    """ロジット関数。シグモイド関数の逆関数。"""
    import keras.backend as K
    x = K.clip(x, K.epsilon(), 1 - K.epsilon())
    return K.log(x / (1 - x))
