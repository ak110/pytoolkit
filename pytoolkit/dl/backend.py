"""keras.backendやtensorflowの基礎的な関数など。"""

import numpy as np


def logit(x):
    """ロジット関数。シグモイド関数の逆関数。

    普通にやると `log(x / (1 - x))` だが、ゼロ除算とlog(0)を避けるためにちょっと面倒な実装に。
    """
    import keras.backend as K
    odds = x / K.maximum(1 - x, K.epsilon())
    return K.log(K.maximum(odds, K.epsilon()))
