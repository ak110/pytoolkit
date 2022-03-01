"""カスタムレイヤー。"""
import numpy as np
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
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


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
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
        shape = tf.shape(inputs)
        int_shape = inputs.shape.as_list()
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


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
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
        outputs.set_shape(inputs.shape.as_list())
        return outputs

    def get_config(self):
        config = {"r": self.r}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
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
        assert input_shape[-1] % (self.scale**2) == 0
        h = None if input_shape[1] is None else input_shape[1] * self.scale
        w = None if input_shape[2] is None else input_shape[2] * self.scale
        return input_shape[0], h, w, input_shape[3] // (self.scale**2)

    def call(self, inputs, **kwargs):
        del kwargs
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

    def get_config(self):
        config = {"scale": self.scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class BlurPooling2D(tf.keras.layers.Layer):
    """Blur Pooling Layer <https://arxiv.org/abs/1904.11486>"""

    def __init__(self, taps=5, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.taps = taps
        self.strides = tk.utils.normalize_tuple(strides, 2)
        self.kernel = None

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

    def build(self, input_shape):
        in_filters = int(input_shape[-1])
        pascals_tr = np.zeros((self.taps, self.taps), dtype=np.float32)
        pascals_tr[0, 0] = 1
        for i in range(1, self.taps):
            pascals_tr[i, :] = pascals_tr[i - 1, :]
            pascals_tr[i, 1:] += pascals_tr[i - 1, :-1]
        filter1d = pascals_tr[self.taps - 1, :]
        filter2d = filter1d[np.newaxis, :] * filter1d[:, np.newaxis]
        filter2d = filter2d * (self.taps**2 / filter2d.sum())
        kernel = np.tile(filter2d[:, :, np.newaxis, np.newaxis], (1, 1, in_filters, 1))
        self.kernel = tf.constant(kernel, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        kernel = tf.cast(self.kernel, inputs.dtype)
        s = tf.shape(inputs)
        outputs = tf.nn.depthwise_conv2d(
            inputs, kernel, strides=(1,) + self.strides + (1,), padding="SAME"
        )
        norm = tf.ones((s[0], s[1], s[2], 1), dtype=inputs.dtype)
        norm = tf.nn.depthwise_conv2d(
            norm,
            kernel[:, :, :1, :],
            strides=(1,) + self.strides + (1,),
            padding="SAME",
        )
        return outputs / norm

    def get_config(self):
        config = {"taps": self.taps, "strides": self.strides}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
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
        x = tf.cast(inputs, tf.float32)  # float16ではオーバーフローしやすいので一応
        x = tf.math.maximum(x, self.epsilon) ** self.p
        x = tf.math.reduce_mean(x, axis=(1, 2))  # GAP
        x = x ** (1 / self.p)
        x = tf.cast(x, inputs.dtype)
        return x

    def get_config(self):
        config = {"p": self.p, "epsilon": self.epsilon}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class GeMPooling2D(tf.keras.layers.Layer):
    """Generalized Mean Pooling (GeM)

    References:
        - <https://arxiv.org/abs/1711.02512>

    """

    def __init__(
        self,
        epsilon=1e-6,
        p_initializer=None,
        p_regularizer=None,
        p_constraint=None,
        p_trainable=True,
        **kargs,
    ):
        super().__init__(**kargs)
        self.epsilon = epsilon
        self.p_initializer = tf.keras.initializers.get(
            p_initializer
            if p_initializer is not None
            else tf.keras.initializers.constant(3)
        )
        self.p_regularizer = tf.keras.regularizers.get(p_regularizer)
        self.p_constraint = tf.keras.constraints.get(p_constraint)
        self.p_trainable = p_trainable
        self.p = None

    def build(self, input_shape):
        self.p = self.add_weight(
            shape=(),
            initializer=self.p_initializer,
            regularizer=self.p_regularizer,
            constraint=self.p_constraint,
            trainable=self.p_trainable,
            name="p",
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        return (input_shape[0], input_shape[3])

    def call(self, inputs, **kwargs):
        del kwargs
        x = tf.cast(inputs, tf.float32)  # float16ではオーバーフローしやすいので一応
        p = tf.cast(self.p, tf.float32)
        p = tf.clip_by_value(p, 1.0, 5.0)  # オーバーフロー対策
        x = tf.math.maximum(x, self.epsilon) ** p
        x = tf.math.reduce_mean(x, axis=(1, 2))  # GAP
        x = x ** (1 / p)
        x = tf.cast(x, inputs.dtype)
        return x

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "p_initializer": tf.keras.initializers.serialize(self.p_initializer),
            "p_regularizer": tf.keras.regularizers.serialize(self.p_regularizer),
            "p_constraint": tf.keras.constraints.serialize(self.p_constraint),
            "p_trainable": self.p_trainable,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
