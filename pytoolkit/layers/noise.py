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


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class DropBlock2D(tf.keras.layers.Layer):
    """DropBlock <https://arxiv.org/abs/1810.12890>"""

    def __init__(
        self, keep_prob: float = 0.9, block_size: int = 7, scale: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.scale = scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        del kwargs

        def drop():
            mask = self._create_mask(input_shape=tf.shape(inputs))
            output = inputs * mask
            if self.scale:
                s = tf.math.reduce_mean(mask, axis=[1, 2, 3], keepdims=True)
                output = output / s
            return output

        outputs = tf.keras.backend.in_train_phase(drop, inputs, training=training)
        return outputs

    def _create_mask(self, input_shape):
        sampling_shape = [
            input_shape[0],
            input_shape[1] - self.block_size + 1,
            input_shape[2] - self.block_size + 1,
            input_shape[3],
        ]
        gamma_n = (1.0 - self.keep_prob) * tf.cast(
            input_shape[2] * input_shape[1], tf.float32
        )
        gamma_d = tf.cast(
            (self.block_size**2)
            * (
                (input_shape[2] - self.block_size + 1)
                * (input_shape[1] - self.block_size + 1)
            ),
            tf.float32,
        )
        gamma = gamma_n / gamma_d
        mask = tf.nn.relu(tf.math.sign(gamma - tf.random.uniform(sampling_shape, 0, 1)))

        p0 = (self.block_size - 1) // 2
        p1 = (self.block_size - 1) - p0
        mask = tf.pad(mask, ((0, 0), (p0, p1), (p0, p1), (0, 0)))
        mask = tf.nn.max_pool(
            mask,
            ksize=(1, self.block_size, self.block_size, 1),
            strides=1,
            padding="SAME",
            data_format="NHWC",
        )
        mask = 1.0 - mask
        return mask

    def get_config(self):
        config = {
            "keep_prob": self.keep_prob,
            "block_size": self.block_size,
            "scale": self.scale,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
