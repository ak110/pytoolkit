"""カスタムレイヤー。"""
import tensorflow as tf

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
class WSConv2D(tf.keras.layers.Conv2D):
    """Weight StandardizationなConv2D <https://arxiv.org/abs/1903.10520>"""

    def call(self, inputs, **kwargs):
        del kwargs

        kernel_mean = K.mean(self.kernel, axis=[0, 1, 2])
        kernel_std = K.std(self.kernel, axis=[0, 1, 2])
        kernel = (self.kernel - kernel_mean) / (kernel_std + 1e-5)

        outputs = K.conv2d(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, self.data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
