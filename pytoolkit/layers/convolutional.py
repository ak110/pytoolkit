"""カスタムレイヤー。"""
import tensorflow as tf

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class CoordChannel1D(tf.keras.layers.Layer):
    """CoordConvなレイヤー。

    ■[1807.03247] An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    """

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 3
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        input_shape = K.shape(inputs)
        pad_shape = (input_shape[0], input_shape[1], 1)
        ones = tf.ones(pad_shape, K.floatx())
        gradation = K.cast(K.arange(0, input_shape[1]), K.floatx()) / K.cast(
            input_shape[1], K.floatx()
        )
        pad_channel = ones * K.reshape(gradation, (1, input_shape[1], 1))
        return K.concatenate([inputs] + [pad_channel], axis=-1)


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class CoordChannel2D(tf.keras.layers.Layer):
    """CoordConvなレイヤー。

    ■[1807.03247] An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    """

    def __init__(self, x_channel=True, y_channel=True, **kwargs):
        super().__init__(**kwargs)
        self.x_channel = x_channel
        self.y_channel = y_channel

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        if self.x_channel:
            input_shape[-1] += 1
        if self.y_channel:
            input_shape[-1] += 1
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        input_shape = K.shape(inputs)
        pad_shape = (input_shape[0], input_shape[1], input_shape[2], 1)
        ones = tf.ones(pad_shape, K.floatx())
        pad_channels = []
        if self.x_channel:
            gradation = K.cast(K.arange(0, input_shape[2]), K.floatx()) / K.cast(
                input_shape[2], K.floatx()
            )
            pad_channels.append(ones * K.reshape(gradation, (1, 1, input_shape[2], 1)))
        if self.y_channel:
            gradation = K.cast(K.arange(0, input_shape[1]), K.floatx()) / K.cast(
                input_shape[1], K.floatx()
            )
            pad_channels.append(ones * K.reshape(gradation, (1, input_shape[1], 1, 1)))
        return K.concatenate([inputs] + pad_channels, axis=-1)

    def get_config(self):
        config = {"x_channel": self.x_channel, "y_channel": self.y_channel}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class WSConv2D(tf.keras.layers.Conv2D):
    """Weight StandardizationなConv2D <https://arxiv.org/abs/1903.10520>"""

    def call(self, inputs):
        # pylint: disable=access-member-before-definition,attribute-defined-outside-init
        base_kernel = self.kernel  # type: ignore
        kernel_mean = tf.math.reduce_mean(base_kernel, axis=[0, 1, 2], keepdims=True)
        kernel_std = tf.math.reduce_std(base_kernel, axis=[0, 1, 2], keepdims=True)
        self.kernel = (base_kernel - kernel_mean) / (kernel_std + 1e-5)
        try:
            return super().call(inputs)
        finally:
            self.kernel = base_kernel


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class CoordEmbedding2D(tf.keras.layers.Layer):
    """CoordConvのようなものでチャンネル数を増やさないようにしてみたもの。"""

    def __init__(self, mirror=True, **kwargs):
        super().__init__(**kwargs)
        self.mirror = mirror

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs
        outputs = tf.identity(inputs)
        s = tf.shape(inputs)
        ex = tf.reshape(tf.linspace(-1.0, +1.0, s[2]), (1, 1, -1, 1))
        if self.mirror:
            ex = tf.math.abs(ex)
        ey = tf.reshape(tf.linspace(-1.0, +1.0, s[1]), (1, -1, 1, 1))
        ex = tf.tile(ex, (1, s[1], 1, s[3] // 2))
        ey = tf.tile(ey, (1, 1, s[2], s[3] - s[3] // 2))
        return outputs + tf.concat([ex, ey], axis=-1)

    def get_config(self):
        config = {"mirror": self.mirror}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
