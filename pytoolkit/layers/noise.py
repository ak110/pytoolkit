"""カスタムレイヤー。"""
import numpy as np
import tensorflow as tf

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class MixFeat(tf.keras.layers.Layer):
    """MixFeat <https://openreview.net/forum?id=HygT9oRqFX>"""

    def __init__(self, sigma=0.2, **kargs):
        super().__init__(**kargs)
        self.sigma = sigma

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        def _passthru():
            return inputs

        def _mixfeat():
            @tf.custom_gradient
            def _forward(x):
                shape = K.shape(x)
                indices = K.arange(start=0, stop=shape[0])
                indices = tf.random.shuffle(indices)
                rs = K.concatenate([K.constant([1], dtype="int32"), shape[1:]])
                r = K.random_normal(rs, 0, self.sigma, dtype="float16")
                theta = K.random_uniform(rs, -np.pi, +np.pi, dtype="float16")
                a = 1 + r * K.cos(theta)
                b = r * K.sin(theta)
                y = x * K.cast(a, K.floatx()) + K.gather(x, indices) * K.cast(
                    b, K.floatx()
                )

                def _backword(dy):
                    inv = tf.math.invert_permutation(indices)
                    return dy * K.cast(a, K.floatx()) + K.gather(dy, inv) * K.cast(
                        b, K.floatx()
                    )

                return y, _backword

            return _forward(inputs)

        return K.in_train_phase(_mixfeat, _passthru, training=training)

    def get_config(self):
        config = {"sigma": self.sigma}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
