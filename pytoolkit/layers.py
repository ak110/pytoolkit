"""Kerasのカスタムレイヤーなど。"""
import numpy as np
import tensorflow as tf

import pytoolkit as tk

from . import utils as tk_utils

K = tf.keras.backend


@tk_utils.register_keras_custom_object
class ConvertColor(tf.keras.layers.Layer):
    """ColorNet <https://arxiv.org/abs/1902.00267> 用の色変換とついでにスケーリング。

    入力は[0, 255]、出力はモード次第だが-3 ～ +3程度。

    Args:
        mode:
            'rgb_to_rgb'
            'rgb_to_lab'
            'rgb_to_hsv'
            'rgb_to_yuv'
            'rgb_to_ycbcr'
            'rgb_to_hed'
            'rgb_to_yiq'
            のいずれか。

    """

    def __init__(self, mode: str, **kargs):
        super().__init__(**kargs)
        self.mode = mode

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs
        if self.mode == "rgb_to_rgb":
            outputs = inputs / 127.5 - 1
        elif self.mode == "rgb_to_lab":
            x = inputs / 255
            mask = K.cast(x > 0.04045, "float32")
            t = mask * K.pow((x + 0.055) / 1.055, 2.4) + (1 - mask) * (x / 12.92)
            m = K.constant(
                np.transpose(
                    [
                        [0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227],
                    ]
                )
            )
            xyz = K.dot(t, m)

            t = xyz / K.constant(
                np.reshape([0.95047, 1.0, 1.08883], (1,) * (K.ndim(inputs) - 1) + (3,))
            )
            mask = K.cast(t > 0.008856, "float32")
            fxfyfz = mask * K.pow(t, 1 / 3) + (1 - mask) * (7.787 * t + 16 / 116)

            x, y, z = fxfyfz[..., 0], fxfyfz[..., 1], fxfyfz[..., 2]
            L = (1.16 * y) - 0.16
            a = 5 * (x - y)
            b = 2 * (y - z)
            outputs = K.stack([L, a, b], axis=-1)
        elif self.mode == "rgb_to_hsv":
            outputs = tf.image.rgb_to_hsv(inputs / 255)
        elif self.mode == "rgb_to_yuv":
            outputs = tf.image.rgb_to_yuv(inputs / 255)
        elif self.mode == "rgb_to_ycbcr":
            m = K.constant(
                np.transpose(
                    [
                        [65.481, 128.553, 24.966],
                        [-37.797, -74.203, 112.0],
                        [112.0, -93.786, -18.214],
                    ]
                )
            )
            b = np.array([16, 128, 128]).reshape((1,) * (K.ndim(inputs) - 1) + (3,))
            outputs = (K.dot(inputs / 255, m) + K.constant(b)) / 255
        elif self.mode == "rgb_to_hed":
            t = inputs / 255 + 2
            m = K.constant(
                [
                    [1.87798274, -1.00767869, -0.55611582],
                    [-0.06590806, 1.13473037, -0.1355218],
                    [-0.60190736, -0.48041419, 1.57358807],
                ]
            )
            outputs = K.dot(-K.log(t) / K.constant(np.log(10)), m)
        elif self.mode == "rgb_to_yiq":
            m = K.constant(
                np.transpose(
                    [
                        [0.299, 0.587, 0.114],
                        [0.59590059, -0.27455667, -0.32134392],
                        [0.21153661, -0.52273617, 0.31119955],
                    ]
                )
            )
            outputs = K.dot(inputs / 255, m)
        else:
            raise ValueError(f"Mode error: {self.mode}")
        return outputs

    def get_config(self):
        config = {"mode": self.mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
class Resize2D(tf.keras.layers.Layer):
    """リサイズ。

    Args:
        size: (new_height, new_width)
        scale: float (sizeと排他でどちらか必須)
        interpolation: 'bilinear', 'nearest', 'bicubic', 'lanczos3', 'lanczos5', 'area'

    """

    def __init__(self, size=None, scale=None, interpolation="bilinear", **kwargs):
        super().__init__(**kwargs)
        assert (size is None) != (scale is None)
        assert interpolation in (
            "bilinear",
            "nearest",
            "bicubic",
            "lanczos3",
            "lanczos5",
            "area",
        )
        self.size = None if size is None else tuple(size)
        self.scale = None if scale is None else float(scale)
        self.interpolation = interpolation

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        if self.size is not None:
            return (input_shape[0], self.size[0], self.size[1], input_shape[-1])
        else:
            new_h = None if input_shape[1] is None else int(input_shape[1] * self.scale)
            new_w = None if input_shape[2] is None else int(input_shape[2] * self.scale)
            return (input_shape[0], new_h, new_w, input_shape[-1])

    def call(self, inputs, **kwargs):
        del kwargs
        method = {
            "bilinear": tf.image.ResizeMethod.BILINEAR,
            "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            "bicubic": tf.image.ResizeMethod.BICUBIC,
            "area": tf.image.ResizeMethod.AREA,
        }[self.interpolation]
        if self.size is not None:
            size = self.size
        else:
            shape = K.shape(inputs)
            scale = K.constant(self.scale, dtype="float32")
            new_h = K.cast(K.cast(shape[1], "float32") * scale, "int32")
            new_w = K.cast(K.cast(shape[2], "float32") * scale, "int32")
            size = (new_h, new_w)
        return tf.image.resize(inputs, size, method=method)

    def get_config(self):
        config = {
            "size": self.size,
            "scale": self.scale,
            "interpolation": self.interpolation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class Pad2D(tf.keras.layers.Layer):
    """tf.padするレイヤー。"""

    def __init__(self, padding=(1, 1), mode="constant", constant_values=0, **kwargs):
        super().__init__(**kwargs)

        assert mode in ("constant", "reflect", "symmetric")

        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        else:
            assert len(padding) == 2
            if isinstance(padding[0], int):
                padding = ((padding[0], padding[0]), (padding[1], padding[1]))
            else:
                assert len(padding[0]) == 2
                assert len(padding[1]) == 2
                padding = (tuple(padding[0]), tuple(padding[1]))

        self.padding = padding
        self.mode = mode
        self.constant_values = constant_values

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[1] += sum(self.padding[0])
        input_shape[2] += sum(self.padding[1])
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        padding = K.constant(((0, 0),) + self.padding + ((0, 0),), dtype="int32")
        return tf.pad(
            tensor=inputs,
            paddings=padding,
            mode=self.mode,
            constant_values=self.constant_values,
            name=self.name,
        )

    def get_config(self):
        config = {
            "padding": self.padding,
            "mode": self.mode,
            "constant_values": self.constant_values,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class PadChannel2D(tf.keras.layers.Layer):
    """チャンネルに対してtf.padするレイヤー。"""

    def __init__(self, filters, mode="constant", constant_values=0, **kwargs):
        super().__init__(**kwargs)
        assert mode in ("constant", "reflect", "symmetric")
        self.filters = filters
        self.mode = mode
        self.constant_values = constant_values

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[3] += self.filters
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        padding = K.constant(((0, 0), (0, 0), (0, 0), (0, self.filters)), dtype="int32")
        return tf.pad(
            tensor=inputs,
            paddings=padding,
            mode=self.mode,
            constant_values=self.constant_values,
            name=self.name,
        )

    def get_config(self):
        config = {
            "filters": self.filters,
            "mode": self.mode,
            "constant_values": self.constant_values,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
class ChannelPair2D(tf.keras.layers.Layer):
    """チャンネル同士の2個の組み合わせの積。"""

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + ((input_shape[-1] * (input_shape[-1] - 1) // 2),)

    def call(self, inputs, **kwargs):
        del kwargs
        ch = K.int_shape(inputs)[-1]
        return K.concatenate(
            [inputs[..., i : i + 1] * inputs[..., i + 1 :] for i in range(ch - 1)],
            axis=-1,
        )


@tk_utils.register_keras_custom_object
class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
    """Sync BN。"""

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        del kwargs
        return K.in_train_phase(
            lambda: self._bn_train(inputs), lambda: self._bn_test(inputs), training
        )

    def _bn_train(self, inputs):
        """学習時のBN。"""
        # self.axisを除く軸で平均・分散を算出する
        target_axis = self.axis
        if isinstance(target_axis, int):
            target_axis = [target_axis]
        stat_axes = [a for a in range(K.ndim(inputs)) if a not in target_axis]

        # 平均・分散の算出
        x = inputs if K.dtype(inputs) == "float32" else K.cast(inputs, "float32")
        mean = K.mean(x, axis=stat_axes)
        squared_mean = K.mean(K.square(x), axis=stat_axes)
        # Sync BN
        if tk.hvd.initialized():
            import horovod.tensorflow as _hvd

            mean = _hvd.allreduce(mean, average=True)
            squared_mean = _hvd.allreduce(squared_mean, average=True)
        var = squared_mean - K.square(mean)

        # exponential moving average:
        # m_new = m_old * 0.99 + x * 0.01
        # m_new - m_old = (x - m_old) * 0.01
        decay = 1 - self.momentum
        update1 = tf.compat.v1.assign_add(
            self.moving_mean, (mean - self.moving_mean) * decay
        )
        update2 = tf.compat.v1.assign_add(
            self.moving_variance, (var - self.moving_variance) * decay
        )
        self.add_update([update1, update2], inputs)

        # y = (x - mean) / (sqrt(var) + epsilon) * gamma + beta
        #   = x * gamma / (sqrt(var) + epsilon) + (beta - mean * gamma / (sqrt(var) + epsilon))
        #   = x * a + (beta - mean * a)
        a = self.gamma / (K.sqrt(var) + 1e-3)
        b = self.beta - mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))

    def _bn_test(self, inputs):
        """予測時のBN。"""
        a = self.gamma / (K.sqrt(self.moving_variance) + 1e-3)
        b = self.beta - self.moving_mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))


@tk_utils.register_keras_custom_object
class GroupNormalization(tf.keras.layers.Layer):
    """Group Normalization。

    Args:
        groups: グループ数

    References:
        - Group Normalization <https://arxiv.org/abs/1803.08494>

    """

    def __init__(
        self,
        groups=32,
        epsilon=1e-5,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        dim = int(input_shape[-1])
        groups = min(dim, self.groups)
        assert dim is None or dim % groups == 0
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs
        x = inputs
        ndim = K.ndim(x)
        shape = K.shape(x)
        if ndim == 4:  # 2D
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            g = K.minimum(self.groups, C)
            x = K.reshape(x, [N, H, W, g, C // g])
            mean, var = tf.compat.v2.nn.moments(x=x, axes=[1, 2, 4], keepdims=True)
            x = (x - mean) / K.sqrt(var + self.epsilon)
            x = K.reshape(x, [N, H, W, C])
        elif ndim == 5:  # 3D
            N, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
            g = K.minimum(self.groups, C)
            x = K.reshape(x, [N, T, H, W, g, C // g])
            mean, var = tf.compat.v2.nn.moments(x=x, axes=[1, 2, 3, 5], keepdims=True)
            x = (x - mean) / K.sqrt(var + self.epsilon)
            x = K.reshape(x, [N, T, H, W, C])
        else:
            assert ndim in (4, 5)
        if self.scale:
            x = x * self.gamma
        if self.center:
            x = x + self.beta
        # tf.keras用
        x.set_shape(K.int_shape(inputs))
        return x

    def get_config(self):
        config = {
            "groups": self.groups,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization"""

    def __init__(
        self,
        epsilon=1e-5,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        if self.scale:
            self.gamma = self.add_weight(
                shape=affine_shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=affine_shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(1, len(input_shape) - 1))
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        std = K.std(inputs, reduction_axes, keepdims=True)
        outputs = (inputs - mean) / (std + self.epsilon)

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[-1] = input_shape[-1]
        if self.scale:
            outputs = outputs * K.reshape(self.gamma, broadcast_shape)
        if self.center:
            outputs = outputs + K.reshape(self.beta, broadcast_shape)

        return outputs

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class RMSNormalization(tf.keras.layers.Layer):
    """Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>"""

    def __init__(
        self,
        axis=-1,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        if self.scale:
            self.gamma = self.add_weight(
                shape=affine_shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=affine_shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs

        ms = tf.math.reduce_mean(inputs ** 2, axis=self.axis, keepdims=True)
        outputs = inputs * tf.math.rsqrt(ms + self.epsilon)

        broadcast_shape = (1,) * (K.ndim(inputs) - 1) + (-1,)
        if self.scale:
            outputs = outputs * tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            outputs = outputs + tf.reshape(self.beta, broadcast_shape)

        return outputs

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class FilterResponseNormalization(tf.keras.layers.Layer):
    """Filter Response Normalization Layer <https://arxiv.org/abs/1911.09737>"""

    def __init__(
        self,
        epsilon=1e-6,
        center=True,
        scale=True,
        activate=True,
        tau_initializer="zeros",
        beta_initializer="zeros",
        gamma_initializer="ones",
        tau_regularizer=None,
        beta_regularizer=None,
        gamma_regularizer=None,
        tau_constraint=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.activate = activate
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.tau = None
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        if self.scale:
            self.gamma = self.add_weight(
                shape=affine_shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        if self.center:
            self.beta = self.add_weight(
                shape=affine_shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        if self.activate:
            self.tau = self.add_weight(
                shape=affine_shape,
                name="tau",
                initializer=self.tau_initializer,
                regularizer=self.tau_regularizer,
                constraint=self.tau_constraint,
            )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        x = inputs
        nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
        x *= tf.math.rsqrt(nu2 + tf.abs(self.epsilon))
        if self.scale:
            x *= self.gamma
        if self.center:
            x += self.beta
        if self.activate:
            x = tf.maximum(x, self.tau)
        return x

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "activate": self.activate,
            "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer),
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "tau_regularizer": tf.keras.regularizers.serialize(self.tau_regularizer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "tau_constraint": tf.keras.constraints.serialize(self.tau_constraint),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class TLU(tf.keras.layers.Layer):
    """Thresholded Linear Unit <https://arxiv.org/abs/1911.09737>"""

    def __init__(
        self,
        tau_initializer="zeros",
        tau_regularizer=None,
        tau_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)
        self.tau = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        self.tau = self.add_weight(
            shape=affine_shape,
            name="tau",
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        return tf.maximum(inputs, self.tau)

    def get_config(self):
        config = {
            "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer),
            "tau_regularizer": tf.keras.regularizers.serialize(self.tau_regularizer),
            "tau_constraint": tf.keras.constraints.serialize(self.tau_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
class DropActivation(tf.keras.layers.Layer):
    """Drop-Activation <https://arxiv.org/abs/1811.05850>"""

    def __init__(self, keep_rate=0.95, **kargs):
        super().__init__(**kargs)
        assert 0 <= keep_rate < 1
        self.keep_rate = keep_rate

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        def _train():
            shape = K.shape(inputs)
            r = K.random_uniform(shape=(shape[0],) + (1,) * (K.ndim(inputs) - 1))
            return tf.compat.v2.where(r <= self.keep_rate, K.relu(inputs), inputs)

        def _test():
            return K.relu(inputs, alpha=1 - self.keep_rate)

        return K.in_train_phase(_train, _test, training=training)

    def get_config(self):
        config = {"keep_rate": self.keep_rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class ParallelGridPooling2D(tf.keras.layers.Layer):
    """Parallel Grid Poolingレイヤー。

    ■ Parallel Grid Pooling for Data Augmentation
    https://arxiv.org/abs/1803.11370

    ■ akitotakeki/pgp-chainer: Chainer Implementation of Parallel Grid Pooling for Data Augmentation
    https://github.com/akitotakeki/pgp-chainer

    """

    def __init__(self, pool_size=(2, 2), **kargs):
        super().__init__(**kargs)
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        assert len(self.pool_size) == 2

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] % self.pool_size[0] == 0  # パディングはとりあえず未対応
        assert input_shape[2] % self.pool_size[1] == 0  # パディングはとりあえず未対応
        b, h, w, c = input_shape
        return b, h // self.pool_size[0], w // self.pool_size[1], c

    def call(self, inputs, **kwargs):
        del kwargs
        shape = K.shape(inputs)
        int_shape = K.int_shape(inputs)
        rh, rw = self.pool_size
        b, h, w, c = shape[0], shape[1], shape[2], int_shape[3]
        outputs = K.reshape(inputs, (b, h // rh, rh, w // rw, rw, c))
        outputs = tf.transpose(a=outputs, perm=(2, 4, 0, 1, 3, 5))
        outputs = K.reshape(outputs, (rh * rw * b, h // rh, w // rw, c))
        # tf.keras用workaround
        if hasattr(outputs, "set_shape"):
            outputs.set_shape(self.compute_output_shape(int_shape))
        return outputs

    def get_config(self):
        config = {"pool_size": self.pool_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class ParallelGridGather(tf.keras.layers.Layer):
    """ParallelGridPoolingでparallelにしたのを戻すレイヤー。"""

    def __init__(self, r, **kargs):
        super().__init__(**kargs)
        self.r = r

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs
        shape = K.shape(inputs)
        b = shape[0]
        gather_shape = K.concatenate([[self.r, b // self.r], shape[1:]], axis=0)
        outputs = K.reshape(inputs, gather_shape)
        outputs = K.mean(outputs, axis=0)
        # tf.keras用workaround
        if hasattr(outputs, "set_shape"):
            outputs.set_shape(K.int_shape(inputs))
        return outputs

    def get_config(self):
        config = {"r": self.r}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class SubpixelConv2D(tf.keras.layers.Layer):
    """Sub-Pixel Convolutional Layer。

    ■ Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    https://arxiv.org/abs/1609.05158

    """

    def __init__(self, scale=2, **kargs):
        super().__init__(**kargs)
        self.scale = scale

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[-1] % (self.scale ** 2) == 0
        h = None if input_shape[1] is None else input_shape[1] * self.scale
        w = None if input_shape[2] is None else input_shape[2] * self.scale
        return input_shape[0], h, w, input_shape[3] // (self.scale ** 2)

    def call(self, inputs, **kwargs):
        del kwargs
        return tf.compat.v1.depth_to_space(input=inputs, block_size=self.scale)

    def get_config(self):
        config = {"scale": self.scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
class BlurPooling2D(tf.keras.layers.Layer):
    """Blur Pooling Layer <https://arxiv.org/abs/1904.11486>"""

    def __init__(self, taps=5, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.taps = taps
        self.strides = tk_utils.normalize_tuple(strides, 2)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[1] = (
            input_shape[1] + int(input_shape[1]) % self.strides[0]
        ) // self.strides[0]
        input_shape[2] = (
            input_shape[2] + int(input_shape[2]) % self.strides[1]
        ) // self.strides[1]
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        in_filters = K.int_shape(inputs)[-1]

        pascals_tr = np.zeros((self.taps, self.taps))
        pascals_tr[0, 0] = 1
        for i in range(1, self.taps):
            pascals_tr[i, :] = pascals_tr[i - 1, :]
            pascals_tr[i, 1:] += pascals_tr[i - 1, :-1]
        filter1d = pascals_tr[self.taps - 1, :]
        filter2d = filter1d[np.newaxis, :] * filter1d[:, np.newaxis]
        filter2d = filter2d * (self.taps ** 2 / filter2d.sum())
        kernel = np.tile(filter2d[:, :, np.newaxis, np.newaxis], (1, 1, in_filters, 1))
        kernel = K.constant(kernel)

        return tf.nn.depthwise_conv2d(
            inputs, kernel, strides=(1,) + self.strides + (1,), padding="SAME"
        )

    def get_config(self):
        config = {"taps": self.taps, "strides": self.strides}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
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


@tk_utils.register_keras_custom_object
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
        output = tf.compat.v2.where(mask, K.ones_like(inputs), inputs)  # nanを1に置き換え
        output = K.dot(output, self.kernel1)
        output = K.relu(output)
        output = K.dot(output, self.kernel2)
        output = K.bias_add(output, self.bias, data_format="channels_last")
        output = tf.compat.v2.where(mask, output, inputs)  # nan以外はinputsを出力
        return output

    def get_config(self):
        config = {"units": self.units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class GeM2D(tf.keras.layers.Layer):
    """Generalized Mean Pooling (GeM) <https://github.com/filipradenovic/cnnimageretrieval-pytorch>"""

    def __init__(self, p=3, epsilon=1e-6, **kargs):
        super().__init__(**kargs)
        self.p = p
        self.epsilon = epsilon

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        return (input_shape[0], input_shape[3])

    def call(self, inputs, **kwargs):
        del kwargs
        x = K.pow(K.maximum(inputs, self.epsilon), self.p)
        x = K.mean(x, axis=[1, 2])  # GAP
        x = K.pow(x, 1 / self.p)
        return x

    def get_config(self):
        config = {"p": self.p, "epsilon": self.epsilon}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional Encodingレイヤー。

    x(i) = pos / pow(10000, 2 * i / depth)
    PE(pos, 2 * i) = sin(x(i))
    PE(pos, 2 * i + 1) = cos(x(i))

    ↑これを入力に足す。

    → 偶数・奇数で分けるのがやや面倒なので、depthの最初半分がsin, 後ろ半分がcosになるようにする
       && depthは偶数前提にしてしまう

    """

    def call(self, inputs, **kwargs):
        del kwargs
        _, max_length, depth = tf.unstack(K.shape(inputs))
        pos = K.cast(tf.range(max_length), K.floatx())
        i = K.cast(tf.range(depth // 2), K.floatx())
        d = K.cast(depth // 2, K.floatx())
        x_i = K.expand_dims(pos, -1) / K.expand_dims(
            K.pow(10000.0, 2.0 * i / d), 0
        )  # (max_length, depth // 2)
        pe0 = K.sin(x_i)
        pe1 = K.cos(x_i)
        pe = K.concatenate([pe0, pe1], axis=-1)  # (max_length, depth)
        pe = K.expand_dims(pe, axis=0)  # (1, max_length, depth)
        return inputs + pe


@tk_utils.register_keras_custom_object
class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head Attetion"""

    def __init__(
        self, units, heads=8, hidden_rate=1.0, drop_rate=0.1, causal=False, **kwargs
    ):
        super().__init__(**kwargs)
        assert units % heads == 0
        self.units = units
        self.heads = heads
        self.hidden_rate = hidden_rate
        self.drop_rate = drop_rate
        self.causal = causal
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.bq = None
        self.bk = None
        self.bv = None

    def compute_output_shape(self, input_shape):
        seq_shape, _ = input_shape
        return (seq_shape[0], seq_shape[1], self.units)

    def build(self, input_shape):
        seq_shape, ctx_shape = input_shape
        output_units = self.units // self.heads
        hidden_units = int(output_units * self.hidden_rate)
        self.Wq = self.add_weight(
            shape=(self.heads, int(seq_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            name="Wq",
        )
        self.Wk = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            name="Wk",
        )
        self.Wv = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), output_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            name="Wv",
        )
        self.bq = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            name="bq",
        )
        self.bk = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            name="bk",
        )
        self.bv = self.add_weight(
            shape=(self.heads, output_units),
            initializer=tf.keras.initializers.zeros(),
            name="bv",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        seq, ctx = inputs

        outputs = []
        for h in range(self.heads):
            # q.shape == (None, seq.shape[1], hidden_units)
            # k.shape == (None, ctx.shape[1], hidden_units)
            # v.shape == (None, ctx.shape[1], output_units)
            q = K.bias_add(K.dot(seq, self.Wq[h]), self.bq[h])
            k = K.bias_add(K.dot(ctx, self.Wk[h]), self.bk[h])
            v = K.bias_add(K.dot(ctx, self.Wv[h]), self.bv[h])
            k = k / np.sqrt(K.int_shape(k)[-1])
            w = K.batch_dot(q, k, axes=(2, 2))  # (None, seq.shape[1], ctx.shape[1])
            if self.causal:
                w_shape = K.shape(w)
                mask_ones = tf.ones(shape=w_shape, dtype="int32")
                row_index = K.cumsum(mask_ones, axis=1)
                col_index = K.cumsum(mask_ones, axis=2)
                causal_mask = K.greater_equal(row_index, col_index)
                w = tf.compat.v2.where(causal_mask, w, K.tile([[[-np.inf]]], w_shape))
            w = K.softmax(w)
            w = K.dropout(w, level=self.drop_rate)  # Attention Dropout
            a = K.batch_dot(w, K.tanh(v), axes=(2, 1))
            # a.shape == (None, seq.shape[1], output_units)
            outputs.append(a)

        outputs = K.concatenate(outputs, axis=-1)
        return outputs

    def get_config(self):
        config = {
            "units": self.units,
            "heads": self.heads,
            "hidden_rate": self.hidden_rate,
            "drop_rate": self.drop_rate,
            "causal": self.causal,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class MultiHeadAttention2D(tf.keras.layers.Layer):
    """Multi-head Attetionの2D版のようなもの。(怪)"""

    def __init__(self, units, heads=8, hidden_rate=1.0, drop_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        assert units % heads == 0
        self.units = units
        self.heads = heads
        self.hidden_rate = hidden_rate
        self.drop_rate = drop_rate
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.bq = None
        self.bk = None
        self.bv = None

    def compute_output_shape(self, input_shape):
        seq_shape, _ = input_shape
        return (seq_shape[0], seq_shape[1], seq_shape[2], self.units)

    def build(self, input_shape):
        seq_shape, ctx_shape = input_shape
        output_units = self.units // self.heads
        hidden_units = int(output_units * self.hidden_rate)
        self.Wq = self.add_weight(
            shape=(self.heads, int(seq_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="Wq",
        )
        self.Wk = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="Wk",
        )
        self.Wv = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), output_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="Wv",
        )
        self.bq = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="bq",
        )
        self.bk = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="bk",
        )
        self.bv = self.add_weight(
            shape=(self.heads, output_units),
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="bv",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        seq, ctx = inputs
        batch_size = K.shape(seq)[0]

        outputs = []
        for h in range(self.heads):
            # q.shape == (None, seq.shape[1], seq.shape[2], hidden_units)
            # k.shape == (None, ctx.shape[1], ctx.shape[2], hidden_units)
            # v.shape == (None, ctx.shape[1], ctx.shape[2], output_units)
            q = K.bias_add(K.dot(seq, self.Wq[h]), self.bq[h])
            k = K.bias_add(K.dot(seq, self.Wk[h]), self.bk[h])
            v = K.bias_add(K.dot(seq, self.Wv[h]), self.bv[h])
            q = K.reshape(q, (batch_size, -1, K.int_shape(k)[-1]))
            k = K.reshape(k, (batch_size, -1, K.int_shape(k)[-1]))
            v = K.reshape(v, (batch_size, -1, K.int_shape(k)[-1]))
            k = k / np.sqrt(K.int_shape(k)[-1])
            w = K.batch_dot(q, k, axes=(2, 2))  # (None, seq.shape[1], ctx.shape[1])
            w = K.softmax(w)
            w = K.dropout(w, level=self.drop_rate)  # Attention Dropout
            a = K.batch_dot(w, K.tanh(v), axes=(2, 1))
            # a.shape == (None, seq.shape[1], output_units)
            outputs.append(a)

        outputs = K.concatenate(outputs, axis=-1)
        output_shape = self.compute_output_shape([K.shape(seq), K.shape(ctx)])
        outputs = K.reshape(outputs, output_shape)
        return outputs

    def get_config(self):
        config = {
            "units": self.units,
            "heads": self.heads,
            "hidden_rate": self.hidden_rate,
            "drop_rate": self.drop_rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
