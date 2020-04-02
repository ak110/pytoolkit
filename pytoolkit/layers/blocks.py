"""複数のレイヤーを組み合わせたブロック。

save/loadのトラブルなどを考えて基本的にFunctional API。

"""

import tensorflow as tf


def se_block(ratio: int = 4):
    """Squeeze-and-Excitation block with swish。<https://arxiv.org/abs/1709.01507>"""

    def layers(x):
        filters = x.shape[-1]
        x_in = x
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(filters // ratio, activation=tf.nn.swish)(x)
        x = tf.keras.layers.Dense(filters, activation="sigmoid")(x)
        x = tf.keras.layers.Reshape((1, 1, filters))(x)
        x = tf.keras.layers.multiply([x_in, x])
        return x

    return layers
