"""tf.keras.backendやtensorflowの基礎的な関数など。"""


def logit(x):
    """ロジット関数。シグモイド関数の逆関数。"""
    import tensorflow as tf
    x = tf.keras.backend.clip(x, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return tf.keras.backend.log(x / (1 - x))
