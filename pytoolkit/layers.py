"""Kerasのカスタムレイヤーなど。"""
import numpy as np
import tensorflow as tf

import pytoolkit as tk

from . import K, keras


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    return {
        name: obj
        for name, obj in globals().items()
        if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)
    }


class ConvertColor(keras.layers.Layer):
    """ColorNet <https://arxiv.org/abs/1902.00267> 用の色変換とついでにスケーリング。

    入力は[0, 255]、出力はモード次第だが-3 ～ +3程度。

    Args:
        mode (str):
            'rgb_to_rgb'
            'rgb_to_lab'
            'rgb_to_hsv'
            'rgb_to_yuv'
            'rgb_to_ycbcr'
            'rgb_to_hed'
            'rgb_to_yiq'
            のいずれか。

    """

    def __init__(self, mode, **kargs):
        super().__init__(**kargs)
        self.mode = mode

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
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


class RemoveMask(keras.layers.Layer):
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


class Conv2DEx(keras.layers.Layer):
    """float16なConv2D+BN+Act。"""

    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        dilation_rate=1,
        center=True,
        scale=True,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (
            (kernel_size,) * 2 if isinstance(kernel_size, int) else kernel_size
        )
        self.strides = (strides,) * 2 if isinstance(strides, int) else strides
        self.dilation_rate = (
            (dilation_rate,) * 2 if isinstance(dilation_rate, int) else dilation_rate
        )
        self.center = center
        self.scale = scale
        self.activation = keras.activations.get(activation)
        self.kernel = None
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variance = None

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[1] = (
            input_shape[1] + input_shape[1] % self.strides[0]
        ) // self.strides[0]
        input_shape[2] = (
            input_shape[2] + input_shape[2] % self.strides[1]
        ) // self.strides[1]
        input_shape[-1] = self.filters
        return tuple(input_shape)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined. Found `None`."
            )

        kernel_shape = self.kernel_size + (int(input_shape[-1]), self.filters)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=keras.initializers.he_uniform(),
            name="kernel",
            regularizer=keras.regularizers.l2(1e-4),
            constraint=None,
            dtype="float32",
        )

        bn_shape = (self.filters,)
        if self.scale:
            self.gamma = self.add_weight(
                shape=bn_shape,
                name="gamma",
                initializer=keras.initializers.ones(),
                regularizer=keras.regularizers.l2(1e-4),
                constraint=None,
                dtype="float32",
            )
        else:
            self.gamma = K.constant(1.0, dtype="float32")
        if self.center:
            self.beta = self.add_weight(
                shape=bn_shape,
                name="beta",
                initializer=keras.initializers.zeros(),
                regularizer=keras.regularizers.l2(1e-4),
                constraint=None,
                dtype="float32",
            )
        else:
            self.beta = K.constant(0.0, dtype="float32")
        self.moving_mean = self.add_weight(
            shape=bn_shape,
            name="moving_mean",
            initializer=keras.initializers.zeros(),
            trainable=False,
            dtype="float32",
        )
        self.moving_variance = self.add_weight(
            shape=bn_shape,
            name="moving_variance",
            initializer=keras.initializers.ones(),
            trainable=False,
            dtype="float32",
        )

        super().build(input_shape)

    def call(self, inputs, training=None):  # pylint: disable=W0221
        if training is None:
            training = K.learning_phase()
        # conv
        outputs = K.conv2d(
            inputs,
            K.cast(self.kernel, K.dtype(inputs)),
            strides=self.strides,
            padding="same",
            data_format="channels_last",
            dilation_rate=self.dilation_rate,
        )
        # bn
        outputs = K.in_train_phase(
            lambda: self._bn_train(outputs), lambda: self._bn_test(outputs), training
        )
        # act
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def _bn_train(self, inputs):
        """学習時のBN。"""
        inputs32 = K.cast(inputs, "float32")
        mean = K.mean(inputs32, axis=[0, 1, 2])
        squared_mean = K.mean(K.square(inputs32), axis=[0, 1, 2])
        # Sync BN
        if tk.hvd.initialized():
            import horovod.tensorflow as _hvd

            mean = _hvd.allreduce(mean, average=True)
            squared_mean = _hvd.allreduce(squared_mean, average=True)
        var = squared_mean - K.square(mean)

        # exponential moving average:
        # m_new = m_old * 0.99 + x * 0.01
        # m_new - m_old = (x - m_old) * 0.01
        update1 = tf.compat.v1.assign_add(
            self.moving_mean, (mean - self.moving_mean) * 0.01
        )
        update2 = tf.compat.v1.assign_add(
            self.moving_variance, (var - self.moving_variance) * 0.01
        )
        self.add_update([update1, update2], inputs)

        # y = (x - mean) / (sqrt(var) + epsilon) * gamma
        a = self.gamma / (K.sqrt(var) + 1e-3)
        b = self.beta - mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))

    def _bn_test(self, inputs):
        """予測時のBN。"""
        # y = (x - mean) / (sqrt(var) + epsilon) * gamma
        a = self.gamma / (K.sqrt(self.moving_variance) + 1e-3)
        b = self.beta - self.moving_mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "dilation_rate": self.dilation_rate,
            "center": self.center,
            "scale": self.scale,
            "activation": keras.activations.serialize(self.activation),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Resize2D(keras.layers.Layer):
    """リサイズ。

    Args:
        size: (new_height, new_width)
        scale: float (sizeと排他でどちらか必須)
        interpolation: 'bilinear', 'nearest', 'bicubic', 'area'

    """

    def __init__(
        self,
        size=None,
        scale=None,
        interpolation="bilinear",
        align_corners=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (size is None) != (scale is None)
        assert interpolation in ("bilinear", "nearest", "bicubic", "area")
        self.size = None if size is None else tuple(size)
        self.scale = None if scale is None else float(scale)
        self.interpolation = interpolation
        self.align_corners = align_corners

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
        return tf.image.resize_images(inputs, size, method, self.align_corners)

    def get_config(self):
        config = {
            "size": self.size,
            "scale": self.scale,
            "interpolation": self.interpolation,
            "align_corners": self.align_corners,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Pad2D(keras.layers.Layer):
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


class PadChannel2D(keras.layers.Layer):
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


class CoordChannel1D(keras.layers.Layer):
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


class CoordChannel2D(keras.layers.Layer):
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


class ChannelPair2D(keras.layers.Layer):
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


class StocasticAdd(keras.layers.Layer):
    """Stochastic Depth <http://arxiv.org/abs/1603.09382>"""

    def __init__(self, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_rate = drop_rate

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[0] == input_shape[1]
        return input_shape[0]

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        del kwargs
        base, residual = inputs

        def _train():
            drop = K.random_binomial((), p=self.drop_rate)
            return K.switch(drop, lambda: base, lambda: base + residual)

        def _test():
            return base + residual * self.drop_rate

        return K.in_train_phase(_train, _test, training)


class BatchNormalization(keras.layers.BatchNormalization):
    """Sync BN。基本的には互換性があるように元のを継承＆同名で。"""

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

        # y = (x - mean) / (sqrt(var) + epsilon) * gamma
        a = self.gamma / (K.sqrt(var) + 1e-3)
        b = self.beta - mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))

    def _bn_test(self, inputs):
        """予測時のBN。"""
        # y = (x - mean) / (sqrt(var) + epsilon) * gamma
        a = self.gamma / (K.sqrt(self.moving_variance) + 1e-3)
        b = self.beta - self.moving_mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))


class GroupNormalization(keras.layers.Layer):
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
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
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
        # tf.keras用workaround
        if hasattr(x, "set_shape"):
            x.set_shape(K.int_shape(inputs))
        return x

    def get_config(self):
        config = {
            "groups": self.groups,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InstanceNormalization(keras.layers.Layer):
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
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
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
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixFeat(keras.layers.Layer):
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


class DropActivation(keras.layers.Layer):
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
            r = K.random_uniform(shape=(shape[0],))
            return tf.where(r <= self.keep_rate, K.relu(inputs), inputs)

        def _test():
            return K.relu(inputs, alpha=1 - self.keep_rate)

        return K.in_train_phase(_train, _test, training=training)

    def get_config(self):
        config = {"keep_rate": self.keep_rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ParallelGridPooling2D(keras.layers.Layer):
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


class ParallelGridGather(keras.layers.Layer):
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


class SubpixelConv2D(keras.layers.Layer):
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


class WSConv2D(keras.layers.Layer):
    """Weight StandardizationなConv2D <https://arxiv.org/abs/1903.10520>"""

    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        use_bias=False,
        activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = tk.utils.normalize_tuple(kernel_size, 2)
        self.strides = tk.utils.normalize_tuple(strides, 2)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[1] = (
            None
            if input_shape[1] is None
            else (input_shape[1] + input_shape[1] % self.strides[0]) // self.strides[0]
        )
        input_shape[2] = (
            None
            if input_shape[2] is None
            else (input_shape[2] + input_shape[2] % self.strides[1]) // self.strides[1]
        )
        input_shape[-1] = self.filters
        return tuple(input_shape)

    def build(self, input_shape):
        in_filters = int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], in_filters, self.filters),
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel",
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=keras.initializers.zeros(),
                name="bias",
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs

        kernel_mean = K.mean(self.kernel, axis=[0, 1, 2])
        kernel_std = K.std(self.kernel, axis=[0, 1, 2])
        kernel = (self.kernel - kernel_mean) / (kernel_std + 1e-5)

        outputs = K.conv2d(inputs, kernel, padding="same", strides=self.strides)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, "channels_last")
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "use_bias": self.use_bias,
            "activation": keras.activations.serialize(self.activation),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OctaveConv2D(keras.layers.Layer):
    """Octave Convolutional Layer <https://arxiv.org/abs/1904.05049>"""

    def __init__(self, filters, strides=1, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.alpha = alpha
        self.strides = tk.utils.normalize_tuple(strides, 2)
        self.filters_l = int(self.filters * self.alpha)
        self.filters_h = self.filters - self.filters_l
        self.kernel_ll = None
        self.kernel_hl = None
        self.kernel_lh = None
        self.kernel_hh = None

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
        assert input_shape[0][1] * 2 == input_shape[1][1]
        assert input_shape[0][2] * 2 == input_shape[1][2]
        input_shape = [list(input_shape[0]), list(input_shape[1])]
        input_shape[0][-1] = self.filters_l
        input_shape[1][-1] = self.filters_h
        return [tuple(input_shape[0]), tuple(input_shape[1])]

    def build(self, input_shape):
        in_filters_l = int(input_shape[0][-1])
        in_filters_h = int(input_shape[1][-1])
        self.kernel_ll = self.add_weight(
            shape=(3, 3, in_filters_l, self.filters_l),
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel_ll",
        )
        self.kernel_hl = self.add_weight(
            shape=(3, 3, in_filters_h, self.filters_l),
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel_hl",
        )
        self.kernel_lh = self.add_weight(
            shape=(3, 3, in_filters_l, self.filters_h),
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel_lh",
        )
        self.kernel_hh = self.add_weight(
            shape=(3, 3, in_filters_h, self.filters_h),
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel_hh",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        input_l, input_h = inputs

        ll = K.conv2d(input_l, self.kernel_ll, padding="same", strides=self.strides)

        hl = K.pool2d(input_h, (2, 2), (2, 2), padding="same", pool_mode="avg")
        hl = K.conv2d(hl, self.kernel_hl, padding="same", strides=self.strides)

        lh = K.conv2d(input_l, self.kernel_lh, padding="same", strides=self.strides)
        lh = K.resize_images(
            lh, 2, 2, data_format="channels_last", interpolation="bilinear"
        )

        hh = K.conv2d(input_h, self.kernel_hh, padding="same", strides=self.strides)

        output_l = ll + hl
        output_h = hh + lh
        return [output_l, output_h]

    def get_config(self):
        config = {"filters": self.filters, "strides": self.strides, "alpha": self.alpha}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BlurPooling2D(keras.layers.Layer):
    """Blur Pooling Layer <https://arxiv.org/abs/1904.11486>"""

    def __init__(self, taps=5, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.taps = taps
        self.strides = tk.utils.normalize_tuple(strides, 2)

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


class ScaleValue(keras.layers.Layer):
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


class ScaleGradient(keras.layers.Layer):
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


class ImputeNaN(keras.layers.Layer):
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
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel1",
        )
        self.kernel2 = self.add_weight(
            shape=(self.units, dim),
            initializer=keras.initializers.he_uniform(),
            regularizer=keras.regularizers.l2(1e-4),
            name="kernel2",
        )
        self.bias = self.add_weight(
            shape=(dim,), initializer=keras.initializers.zeros(), name="bias"
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


class SoftGate(keras.layers.Layer):
    """LIP用の層。sigmoidして定数倍するだけ。"""

    def __init__(self, coeff=12.0, **kwargs):
        super().__init__(**kwargs)
        self.coeff = coeff

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs
        return K.sigmoid(inputs) * self.coeff

    def get_config(self):
        config = {"coeff": self.coeff}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LIP2D(keras.layers.Layer):
    """LIP <https://arxiv.org/abs/1908.04156>"""

    def __init__(self, pool_size=3, strides=2, padding="same", **kargs):
        super().__init__(**kargs)
        self.pool_size = tk.utils.normalize_tuple(pool_size, 2)
        self.strides = tk.utils.normalize_tuple(strides, 2)
        self.padding = padding

    def compute_output_shape(self, input_shape):
        return keras.layers.AveragePooling2D(
            pool_size=self.pool_size, strides=self.strides, padding=self.padding
        ).compute_output_shape(input_shape[0])

    def call(self, inputs, **kwargs):
        del kwargs
        x, logit = inputs

        weights = K.exp(logit)
        x = x * weights
        x = K.pool2d(
            x,
            self.pool_size,
            self.strides,
            self.padding,
            "channels_last",
            pool_mode="avg",
        )
        weights = K.pool2d(
            weights,
            self.pool_size,
            self.strides,
            self.padding,
            "channels_last",
            pool_mode="avg",
        )
        outputs = x / weights

        return outputs

    def get_config(self):
        config = {
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BottleneckLIP2D(keras.models.Model):
    """LIP <https://arxiv.org/abs/1908.04156>"""
