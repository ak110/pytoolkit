"""カスタムレイヤー。"""
import typing

import numpy as np
import tensorflow as tf

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class RemoveMask(tf.keras.layers.Layer):
    """マスクを取り除く。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):  # pylint: disable=useless-return
        _ = inputs, mask
        return None

    def call(self, inputs, **kwargs):
        _ = kwargs
        return inputs


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class ChannelPair2D(tf.keras.layers.Layer):
    """チャンネル同士の2個の組み合わせの積。"""

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + ((input_shape[-1] * (input_shape[-1] - 1) // 2),)

    def call(self, inputs, **kwargs):
        del kwargs
        ch = inputs.shape[-1]
        return K.concatenate(
            [inputs[..., i : i + 1] * inputs[..., i + 1 :] for i in range(ch - 1)],
            axis=-1,
        )


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class ScaleValue(tf.keras.layers.Layer):
    """値だけをスケーリングしてシフトするレイヤー。回帰の出力前とかに。"""

    def __init__(self, scale, shift=0, **kargs):
        super().__init__(**kargs)
        self.scale = np.float32(scale)
        self.shift = np.float32(shift)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs

        @tf.custom_gradient
        def _forward(x):
            def _backword(dy):
                return dy

            return x * self.scale + self.shift, _backword

        return _forward(inputs)

    def get_config(self):
        config = {"scale": self.scale, "shift": self.shift}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class ScaleGradient(tf.keras.layers.Layer):
    """勾配だけをスケーリングするレイヤー。転移学習するときとかに。"""

    def __init__(self, scale, **kargs):
        super().__init__(**kargs)
        self.scale = np.float32(scale)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs

        @tf.custom_gradient
        def _forward(x):
            def _backword(dy):
                return dy * self.scale

            return x, _backword

        return _forward(inputs)

    def get_config(self):
        config = {"scale": self.scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class ImputeNaN(tf.keras.layers.Layer):
    """NaNを適当な値に変換する層。"""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel1 = None
        self.kernel2 = None
        self.bias = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.kernel1 = self.add_weight(
            shape=(dim, self.units),
            initializer=tf.keras.initializers.he_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="kernel1",
        )
        self.kernel2 = self.add_weight(
            shape=(self.units, dim),
            initializer=tf.keras.initializers.he_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="kernel2",
        )
        self.bias = self.add_weight(
            shape=(dim,), initializer=tf.keras.initializers.zeros(), name="bias"
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        mask = tf.math.is_nan(inputs)
        output = tf.where(mask, K.ones_like(inputs), inputs)  # nanを1に置き換え
        output = K.dot(output, self.kernel1)
        output = K.relu(output)
        output = K.dot(output, self.kernel2)
        output = K.bias_add(output, self.bias, data_format="channels_last")
        output = tf.where(mask, output, inputs)  # nan以外はinputsを出力
        return output

    def get_config(self):
        config = {"units": self.units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class TrainOnly(tf.keras.layers.Wrapper):
    """訓練時のみ適用するlayer wrapper"""

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        return K.in_train_phase(
            lambda: self.layer.call(inputs, **kwargs), inputs, training
        )


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class TestOnly(tf.keras.layers.Wrapper):
    """推論時のみ適用するlayer wrapper"""

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        return K.in_train_phase(
            inputs, lambda: self.layer.call(inputs, **kwargs), training
        )


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class Scale(tf.keras.layers.Layer):
    """学習可能なスケール値。"""

    def __init__(
        self,
        shape=(),
        scale_initializer="zeros",
        scale_regularizer=None,
        scale_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.shape = shape
        self.scale_initializer = tf.keras.initializers.get(scale_initializer)
        self.scale_regularizer = tf.keras.regularizers.get(scale_regularizer)
        self.scale_constraint = tf.keras.constraints.get(scale_constraint)
        self.scale = None

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=self.shape,
            name="scale",
            initializer=self.scale_initializer,
            regularizer=self.scale_regularizer,
            constraint=self.scale_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        return inputs * self.scale

    def get_config(self):
        config = {
            "scale_initializer": tf.keras.initializers.serialize(
                self.scale_initializer
            ),
            "scale_regularizer": tf.keras.regularizers.serialize(
                self.scale_regularizer
            ),
            "scale_constraint": tf.keras.constraints.serialize(self.scale_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class RandomScale(tf.keras.layers.Layer):
    """ランダムにスケーリング

    Args:
        min_scale: 最小値
        max_scale: 最大値
        shape: スケールのshape

    """

    def __init__(
        self,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        shape: typing.Tuple[int, ...] = (),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.shape = shape

    def call(self, inputs, **kwargs):
        del kwargs
        scale = tf.math.exp(
            tf.random.uniform(
                self.shape, np.log(self.min_scale), np.log(self.max_scale)
            )
        )
        return inputs * scale

    def get_config(self):
        config = {
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "shape": self.shape,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
