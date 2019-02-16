"""Kerasのカスタムレイヤーなど。"""

import numpy as np
import tensorflow as tf

from . import K, keras


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = [
        Preprocess,
        Resize2D,
        Pad2D,
        PadChannel2D,
        CoordChannel2D,
        ChannelPair2D,
        GroupNormalization,
        MixFeat,
        DropActivation,
        ParallelGridPooling2D,
        ParallelGridGather,
        SubpixelConv2D,
    ]
    return {c.__name__: c for c in classes}


class Preprocess(keras.layers.Layer):
    """前処理レイヤー。"""

    def __init__(self, mode='tf', **kwargs):
        super().__init__(**kwargs)
        assert mode in ('caffe', 'tf', 'torch', 'div255', 'std')
        self.mode = mode

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        _ = kwargs  # noqa
        if self.mode == 'caffe':
            return K.bias_add(inputs[..., ::-1], K.constant(np.array([-103.939, -116.779, -123.68])))
        elif self.mode == 'tf':
            return (inputs / 127.5) - 1
        elif self.mode == 'torch':
            return K.bias_add((inputs / 255.), K.constant(np.array([-0.485, -0.456, -0.406]))) / np.array([0.229, 0.224, 0.225])
        elif self.mode == 'div255':
            return inputs / 255.
        elif self.mode == 'std':
            axes = tuple(range(1, K.ndim(inputs)))
            return (inputs - K.mean(inputs, axis=axes, keepdims=True)) / (K.std(inputs, axis=axes, keepdims=True) + K.epsilon())
        else:
            assert False
            return None

    def get_config(self):
        config = {'mode': self.mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Resize2D(keras.layers.Layer):
    """リサイズ。

    Args:
        size: (new_height, new_width)
        scale: float (sizeと排他でどちらか必須)
        interpolation: 'bilinear', 'nearest', 'bicubic', 'area'

    """

    def __init__(self, size=None, scale=None, interpolation='bilinear', align_corners=False, **kwargs):
        super().__init__(**kwargs)
        assert (size is None) != (scale is None)
        assert interpolation in ('bilinear', 'nearest', 'bicubic', 'area')
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
        _ = kwargs  # noqa
        method = {
            'bilinear': tf.image.ResizeMethod.BILINEAR,
            'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            'bicubic': tf.image.ResizeMethod.BICUBIC,
            'area': tf.image.ResizeMethod.AREA,
        }[self.interpolation]
        if self.size is not None:
            size = self.size
        else:
            shape = keras.backend.shape(inputs)
            scale = keras.backend.constant(self.scale, dtype='float32')
            new_h = keras.backend.cast(keras.backend.cast(shape[1], 'float32') * scale, 'int32')
            new_w = keras.backend.cast(keras.backend.cast(shape[2], 'float32') * scale, 'int32')
            size = (new_h, new_w)
        return tf.image.resize_images(inputs, size, method, self.align_corners)

    def get_config(self):
        config = {
            'size': self.size,
            'scale': self.scale,
            'interpolation': self.interpolation,
            'align_corners': self.align_corners,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Pad2D(keras.layers.Layer):
    """tf.padするレイヤー。"""

    def __init__(self, padding=(1, 1), mode='constant', constant_values=0, **kwargs):
        super().__init__(**kwargs)

        assert mode in ('constant', 'reflect', 'symmetric')

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
        _ = kwargs  # noqa
        padding = K.constant(((0, 0),) + self.padding + ((0, 0),), dtype='int32')
        return tf.pad(inputs, padding, mode=self.mode, constant_values=self.constant_values, name=self.name)

    def get_config(self):
        config = {
            'padding': self.padding,
            'mode': self.mode,
            'constant_values': self.constant_values,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PadChannel2D(keras.layers.Layer):
    """チャンネルに対してtf.padするレイヤー。"""

    def __init__(self, filters, mode='constant', constant_values=0, **kwargs):
        assert mode in ('constant', 'reflect', 'symmetric')
        super().__init__(**kwargs)
        self.filters = filters
        self.mode = mode
        self.constant_values = constant_values

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 4
        input_shape = list(input_shape)
        input_shape[3] += self.filters
        return tuple(input_shape)

    def call(self, inputs, **kwargs):
        padding = K.constant(((0, 0), (0, 0), (0, 0), (0, self.filters)), dtype='int32')
        return tf.pad(inputs, padding, mode=self.mode, constant_values=self.constant_values, name=self.name)

    def get_config(self):
        config = {
            'filters': self.filters,
            'mode': self.mode,
            'constant_values': self.constant_values,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        _ = kwargs  # noqa
        input_shape = K.shape(inputs)
        pad_shape = (input_shape[0], input_shape[1], input_shape[2], 1)
        ones = tf.ones(pad_shape, K.floatx())
        pad_channels = []
        if self.x_channel:
            gradation = K.cast(K.arange(0, input_shape[2]), K.floatx()) / K.cast(input_shape[2], K.floatx())
            pad_channels.append(ones * K.reshape(gradation, (1, 1, input_shape[2], 1)))
        if self.y_channel:
            gradation = K.cast(K.arange(0, input_shape[1]), K.floatx()) / K.cast(input_shape[1], K.floatx())
            pad_channels.append(ones * K.reshape(gradation, (1, input_shape[1], 1, 1)))
        return K.concatenate([inputs] + pad_channels, axis=-1)

    def get_config(self):
        config = {
            'x_channel': self.x_channel,
            'y_channel': self.y_channel,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelPair2D(keras.layers.Layer):
    """チャンネル同士の2個の組み合わせの積。"""

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + ((input_shape[-1] * (input_shape[-1] - 1) // 2),)

    def call(self, inputs, **kwargs):
        _ = kwargs  # noqa
        ch = K.int_shape(inputs)[-1]
        return K.concatenate([inputs[..., i:i + 1] * inputs[..., i + 1:] for i in range(ch - 1)], axis=-1)


class GroupNormalization(keras.layers.Layer):
    """Group normalization。

    Args:
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    References:
        - Group Normalization <https://arxiv.org/abs/1803.08494>

    """

    def __init__(self,
                 groups=32,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
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
        dim = input_shape[-1]
        assert dim is None or dim % self.groups == 0
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        _ = kwargs  # noqa
        x = inputs
        ndim = K.ndim(x)
        shape = K.shape(x)
        if ndim == 4:  # 2D
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            g = K.minimum(self.groups, C)
            x = K.reshape(x, [N, H, W, g, C // g])
            mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
            x = (x - mean) / K.sqrt(var + self.epsilon)
            x = K.reshape(x, [N, H, W, C])
        elif ndim == 5:  # 3D
            N, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
            g = K.minimum(self.groups, C)
            x = K.reshape(x, [N, T, H, W, g, C // g])
            mean, var = tf.nn.moments(x, [1, 2, 3, 5], keep_dims=True)
            x = (x - mean) / K.sqrt(var + self.epsilon)
            x = K.reshape(x, [N, T, H, W, C])
        else:
            assert ndim in (4, 5)
        if self.scale:
            x = x * self.gamma
        if self.center:
            x = x + self.beta
        return x

    def get_config(self):
        config = {
            'groups': self.groups,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MixFeat(keras.layers.Layer):
    """MixFeat <https://openreview.net/forum?id=HygT9oRqFX>"""

    def __init__(self, sigma=0.2, **kargs):
        self.sigma = sigma
        super().__init__(**kargs)

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
                indices = tf.random_shuffle(indices)
                rs = K.concatenate([K.constant([1], dtype='int32'), shape[1:]])
                r = K.random_normal(rs, 0, self.sigma, dtype='float16')
                theta = K.random_uniform(rs, -np.pi, +np.pi, dtype='float16')
                a = 1 + r * K.cos(theta)
                b = r * K.sin(theta)
                y = x * K.cast(a, K.floatx()) + K.gather(x, indices) * K.cast(b, K.floatx())

                def _backword(dx):
                    inv = tf.invert_permutation(indices)
                    return dx * K.cast(a, K.floatx()) + K.gather(dx, inv) * K.cast(b, K.floatx())

                return y, _backword

            return _forward(inputs)

        return K.in_train_phase(_mixfeat, _passthru, training=training)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DropActivation(keras.layers.Layer):
    """Drop-Activation <https://arxiv.org/abs/1811.05850>"""

    def __init__(self, keep_rate=0.95, **kargs):
        assert 0 <= keep_rate < 1
        self.keep_rate = keep_rate
        super().__init__(**kargs)

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
        config = {'keep_rate': self.keep_rate}
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
        _ = kwargs  # noqa
        shape = keras.backend.shape(inputs)
        int_shape = keras.backend.int_shape(inputs)
        rh, rw = self.pool_size
        b, h, w, c = shape[0], shape[1], shape[2], int_shape[3]
        inputs = keras.backend.reshape(inputs, (b, h // rh, rh, w // rw, rw, c))
        inputs = tf.transpose(inputs, perm=(2, 4, 0, 1, 3, 5))
        inputs = keras.backend.reshape(inputs, (rh * rw * b, h // rh, w // rw, c))
        return inputs

    def get_config(self):
        config = {'pool_size': self.pool_size}
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
        _ = kwargs  # noqa
        shape = keras.backend.shape(inputs)
        b = shape[0]
        gather_shape = keras.backend.concatenate([[self.r, b // self.r], shape[1:]], axis=0)
        inputs = keras.backend.reshape(inputs, gather_shape)
        inputs = keras.backend.mean(inputs, axis=0)
        return inputs

    def get_config(self):
        config = {'r': self.r}
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
        _ = kwargs  # noqa
        return tf.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
