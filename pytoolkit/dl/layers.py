"""Kerasのカスタムレイヤーなど。"""

import numpy as np


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = [
        preprocess(),
        channel_argmax(),
        channel_max(),
        resize2d(),
        pad2d(),
        pad_channel_2d(),
        coord_channel_2d(),
        channel_pair_2d(),
        group_normalization(),
        destandarization(),
        stocastic_add(),
        mixfeat(),
        drop_activation(),
        normal_noise(),
        l2normalization(),
        weighted_mean(),
        parallel_grid_pooling_2d(),
        parallel_grid_gather(),
        subpixel_conv2d(),
        nms(),
    ]
    return {c.__name__: c for c in classes}


def preprocess():
    """前処理レイヤー。"""
    import keras
    import keras.backend as K

    class Preprocess(keras.layers.Layer):
        """前処理レイヤー。"""

        def __init__(self, mode='tf', **kwargs):
            super().__init__(**kwargs)
            assert mode in ('caffe', 'tf', 'torch', 'div255')
            self.mode = mode

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, inputs, **kwargs):
            if self.mode == 'caffe':
                return K.bias_add(inputs[..., ::-1], K.constant(np.array([-103.939, -116.779, -123.68])))
            elif self.mode == 'tf':
                return (inputs / 127.5) - 1
            elif self.mode == 'torch':
                return K.bias_add((inputs / 255.), K.constant(np.array([-0.485, -0.456, -0.406]))) / np.array([0.229, 0.224, 0.225])
            elif self.mode == 'div255':
                return inputs / 255.
            else:
                assert False
                return None

        def get_config(self):
            config = {'mode': self.mode}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return Preprocess


def channel_argmax():
    """チャンネルをargmaxするレイヤー。"""
    import keras
    import keras.backend as K

    class ChannelArgMax(keras.layers.Layer):
        """チャンネルをargmaxするレイヤー。"""

        def compute_output_shape(self, input_shape):
            return input_shape[:-1]

        def call(self, inputs, **kwargs):
            return K.argmax(inputs, axis=-1)

    return ChannelArgMax


def channel_max():
    """チャンネルをmaxするレイヤー。"""
    import keras
    import keras.backend as K

    class ChannelMax(keras.layers.Layer):
        """チャンネルをmaxするレイヤー。"""

        def compute_output_shape(self, input_shape):
            return input_shape[:-1]

        def call(self, inputs, **kwargs):
            return K.max(inputs, axis=-1)

    return ChannelMax


def resize2d():
    """リサイズ。

    Args:
        size: (new_height, new_width)
        scale: float (sizeと排他でどちらか必須)
        interpolation: 'bilinear', 'nearest', 'bicubic', 'area'

    """
    import keras
    import tensorflow as tf

    class Resize2D(keras.layers.Layer):
        """リサイズ。"""

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

    return Resize2D


def pad2d():
    """tf.padするレイヤー。"""
    import keras
    import keras.backend as K
    import tensorflow as tf

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

    return Pad2D


def pad_channel_2d():
    """チャンネルに対してtf.padするレイヤー。"""
    import keras
    import keras.backend as K
    import tensorflow as tf

    class PadChannel2D(keras.layers.Layer):
        """tf.padするレイヤー。"""

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

    return PadChannel2D


def coord_channel_2d():
    """CoordConvなレイヤー。

    ■[1807.03247] An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    """
    import keras
    import keras.backend as K
    import tensorflow as tf

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

    return CoordChannel2D


def channel_pair_2d():
    """チャンネル同士の2個の組み合わせの積。"""
    import keras
    import keras.backend as K

    class ChannelPair2D(keras.layers.Layer):
        """チャンネル同士の2個の組み合わせの積。"""

        def compute_output_shape(self, input_shape):
            return input_shape[:-1] + ((input_shape[-1] * (input_shape[-1] - 1) // 2),)

        def call(self, inputs, **kwargs):
            ch = K.int_shape(inputs)[-1]
            return K.concatenate([inputs[..., i:i + 1] * inputs[..., i + 1:] for i in range(ch - 1)], axis=-1)

    return ChannelPair2D


def group_normalization():
    """Group normalization。"""
    import keras
    import keras.backend as K
    import tensorflow as tf

    class GroupNormalization(keras.layers.Layer):
        """Group normalization。

        # Arguments
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

        # Input shape
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        # Output shape
            Same shape as input.

        # References
            - [Group Normalization](https://arxiv.org/abs/1803.08494)
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
            return x * self.gamma + self.beta

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

    return GroupNormalization


def destandarization():
    """事前に求めた平均と標準偏差を元に出力を標準化するレイヤー。"""
    import keras
    import keras.backend as K

    class Destandarization(keras.layers.Layer):
        """事前に求めた平均と標準偏差を元に出力を標準化するレイヤー。

        # Arguments
            - mean: 平均 (float).
            - std: 標準偏差 (positive float).

        # Input shape
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        # Output shape
            Same shape as input.
        """

        def __init__(self, mean=0, std=0.3, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.mean = K.cast_to_floatx(mean)
            self.std = K.cast_to_floatx(std)
            if self.std <= K.epsilon():
                self.std = 1.  # 怪しい安全装置

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, inputs, **kwargs):
            return inputs * self.std + self.mean

        def get_config(self):
            config = {'mean': float(self.mean), 'std': float(self.std)}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return Destandarization


def stocastic_add():
    """Stocastic Depthのための確率的な加算。"""
    import keras
    import keras.backend as K

    class StocasticAdd(keras.layers.Layer):
        """Stocastic Depthのための確率的な加算。

        Args:
        - p: survival probability。1だとdropせず、0.5だと1/2の確率でdrop

        """

        def __init__(self, survival_prob, calibration=True, **kargs):
            assert 0 < survival_prob <= 1
            self.survival_prob = float(survival_prob)
            self.calibration = calibration
            super().__init__(**kargs)

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            assert input_shape[0] == input_shape[1]
            return input_shape[0]

        def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
            assert len(inputs) == 2

            if self.survival_prob >= 1:
                return inputs[0] + inputs[1]

            def _add():
                return inputs[0] + inputs[1]

            def _drop():
                return inputs[0]

            def _stocastic_add():
                r = K.random_uniform((1,), 0, 1)[0]
                return K.switch(K.less_equal(r, self.survival_prob), _add, _drop)

            def _calibrated_add():
                if self.calibration:
                    return inputs[0] + inputs[1] * self.survival_prob
                else:
                    return inputs[0] + inputs[1]

            return K.in_train_phase(_stocastic_add, _calibrated_add, training=training)

        def get_config(self):
            config = {
                'survival_prob': self.survival_prob,
                'calibration': self.calibration,
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return StocasticAdd


def mixfeat():
    """MixFeat <https://openreview.net/forum?id=HygT9oRqFX>"""
    import keras
    import keras.backend as K
    import tensorflow as tf

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

    return MixFeat


def drop_activation():
    """Drop-Activation <https://arxiv.org/abs/1811.05850>"""
    import keras
    import keras.backend as K
    import tensorflow as tf

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

    return DropActivation


def normal_noise():
    """平均0、分散1のノイズをドロップアウト風に適用する。"""
    import keras
    import keras.backend as K

    class NormalNoise(keras.layers.Layer):
        """平均0、分散1のノイズをドロップアウト風に適用する。"""

        def __init__(self, noise_rate=0.25, **kargs):
            assert 0 <= noise_rate < 1
            self.noise_rate = noise_rate
            super().__init__(**kargs)

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, inputs, training=None):  # pylint: disable=arguments-differ
            if self.noise_rate <= 0:
                return inputs

            def _passthru():
                return inputs

            def _erase_random():
                shape = K.shape(inputs)
                noise = K.random_normal(shape)
                mask = K.cast(K.greater_equal(K.random_uniform(shape), self.noise_rate), K.floatx())
                return inputs * mask + noise * (1 - mask)

            return K.in_train_phase(_erase_random, _passthru, training=training)

        def get_config(self):
            config = {'noise_rate': self.noise_rate}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NormalNoise


def l2normalization():
    """L2 Normalizationレイヤー。"""
    import keras
    import keras.backend as K

    class L2Normalization(keras.layers.Layer):
        """L2 Normalizationレイヤー。"""

        def __init__(self, scale=1, **kargs):
            self.scale = scale
            self.gamma = None
            super().__init__(**kargs)

        def build(self, input_shape):
            ch_axis = 3 if K.image_data_format() == 'channels_last' else 1
            shape = (input_shape[ch_axis],)
            init_gamma = self.scale * np.ones(shape)
            self.gamma = self.add_weight(name='gamma',
                                         shape=shape,
                                         initializer=keras.initializers.Constant(init_gamma),
                                         trainable=True)
            return super().build(input_shape)

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, inputs, **kwargs):
            ch_axis = 3 if K.image_data_format() == 'channels_last' else 1
            output = K.l2_normalize(inputs, ch_axis)
            output *= self.gamma
            return output

        def get_config(self):
            config = {'scale': self.scale}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return L2Normalization


def weighted_mean():
    """入力の加重平均を取るレイヤー。"""
    import keras
    import keras.backend as K

    class WeightedMean(keras.layers.Layer):
        """入力の加重平均を取るレイヤー。"""

        def __init__(self,
                     kernel_initializer=keras.initializers.Constant(0.1),
                     kernel_regularizer=None,
                     kernel_constraint='non_neg',
                     **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.kernel = None
            self.kernel_initializer = keras.initializers.get(kernel_initializer)
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
            self.kernel_constraint = keras.constraints.get(kernel_constraint)

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(len(input_shape),),
                                          name='kernel',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            super().build(input_shape)

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            return input_shape[0]

        def call(self, inputs, **kwargs):
            ot = K.zeros_like(inputs[0])
            for i, inp in enumerate(inputs):
                ot += inp * self.kernel[i]
            ot /= K.sum(self.kernel) + K.epsilon()
            return ot

        def get_config(self):
            config = {
                'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
                'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return WeightedMean


def parallel_grid_pooling_2d():
    """Parallel Grid Poolingレイヤー。

    ■ Parallel Grid Pooling for Data Augmentation
    https://arxiv.org/abs/1803.11370

    ■ akitotakeki/pgp-chainer: Chainer Implementation of Parallel Grid Pooling for Data Augmentation
    https://github.com/akitotakeki/pgp-chainer

    """
    import keras
    import tensorflow as tf

    class ParallelGridPooling2D(keras.layers.Layer):
        """Parallel Grid Poolingレイヤー。"""

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

    return ParallelGridPooling2D


def parallel_grid_gather():
    """ParallelGridPoolingでparallelにしたのを戻すレイヤー。"""
    import keras

    class ParallelGridGather(keras.layers.Layer):
        """ParallelGridPoolingでparallelにしたのを戻すレイヤー。"""

        def __init__(self, r, **kargs):
            super().__init__(**kargs)
            self.r = r

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, inputs, **kwargs):
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

    return ParallelGridGather


def subpixel_conv2d():
    """Sub-Pixel Convolutional Layer。

    ■ Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    https://arxiv.org/abs/1609.05158

    """
    import keras
    import tensorflow as tf

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
            return tf.depth_to_space(inputs, self.scale)

        def get_config(self):
            config = {'scale': self.scale}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return SubpixelConv2D


def nms():
    """Non maximum suppressionを行うレイヤー。

    入力は、classes, confs, locs。
    出力は、画像数×top_k×6
    6はclass(1) + conf(1) + loc(4)。

    出力がtop_k個未満ならzero padding。(confidenceが0になるのでそこそこ無害のはず)

    nms_all_thresholdを指定すると別クラス間でもNMSをする。
    Noneにするとしない。(mAP算出など用)
    """
    import keras
    import keras.backend as K
    import tensorflow as tf

    class NMS(keras.layers.Layer):
        """Non maximum suppressionを行うレイヤー。"""

        def __init__(self, num_classes, prior_boxes, top_k=200, conf_threshold=0.01, nms_threshold=0.45, nms_all_threshold=None, **kwargs):
            super().__init__(**kwargs)
            self.num_classes = num_classes
            self.prior_boxes = prior_boxes
            self.top_k = top_k
            self.conf_threshold = conf_threshold
            self.nms_threshold = nms_threshold
            self.nms_all_threshold = nms_all_threshold

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            return (input_shape[0][0], self.top_k, 1 + 1 + 4)

        def call(self, inputs, **kwargs):
            classes, confs, locs = inputs
            classes = K.reshape(K.cast(classes, K.floatx()), (-1, self.prior_boxes, 1))
            confs = K.reshape(confs, (-1, self.prior_boxes, 1))
            inputs = K.concatenate([classes, confs, locs], axis=-1)
            return K.map_fn(self._process_image, inputs)

        def get_config(self):
            config = {
                'num_classes': self.num_classes,
                'prior_boxes': self.prior_boxes,
                'top_k': self.top_k,
                'conf_threshold': self.conf_threshold,
                'nms_threshold': self.nms_threshold,
                'nms_all_threshold': self.nms_all_threshold,
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

        def _process_image(self, image):
            classes = K.cast(image[:, 0], 'int32')
            confs = image[:, 1]
            locs = image[:, 2:]

            # tf 1.11以降用のworkaround <https://github.com/tensorflow/tensorflow/issues/22581>
            if tf.__version__ in ('1.11.0', '1.12.0'):
                from tensorflow.python.ops import gen_image_ops
                non_max_suppression = gen_image_ops.non_max_suppression_v2
            else:
                non_max_suppression = tf.image.non_max_suppression

            img_classes = []
            img_confs = []
            img_locs = []
            for class_id in range(self.num_classes):
                # 対象のクラスのみ抜き出す
                mask = tf.logical_and(K.equal(classes, class_id), K.greater(confs, self.conf_threshold))
                target_classes = tf.boolean_mask(classes, mask, axis=0)  # 無駄だが、tileの使い方が分からないのでとりあえず。。
                target_confs = tf.boolean_mask(confs, mask, axis=0)
                target_locs = tf.boolean_mask(locs, mask, axis=0)
                # NMS
                top_k = self.top_k * 2 if self.nms_all_threshold else self.top_k  # 全体でもNMSするなら余裕を持たせる必要がある
                top_k = K.minimum(top_k, K.shape(target_confs)[0])
                nms_indices = non_max_suppression(target_locs, target_confs, top_k, self.nms_threshold)
                img_classes.append(K.gather(target_classes, nms_indices))
                img_confs.append(K.gather(target_confs, nms_indices))
                img_locs.append(K.gather(target_locs, nms_indices))

            # 全クラス分くっつける
            img_classes = K.concatenate(img_classes, axis=0)
            img_confs = K.concatenate(img_confs, axis=0)
            img_locs = K.concatenate(img_locs, axis=0)
            # 全クラスで再度NMS or top_k
            top_k = K.minimum(self.top_k, K.shape(img_confs)[0])
            if self.nms_all_threshold:
                nms_indices = non_max_suppression(img_locs, img_confs, top_k, self.nms_all_threshold)
                img_classes = K.gather(img_classes, nms_indices)
                img_confs = K.gather(img_confs, nms_indices)
                img_locs = K.gather(img_locs, nms_indices)
            else:
                img_confs, top_k_indices = tf.nn.top_k(img_confs, top_k, sorted=True)
                img_classes = K.gather(img_classes, top_k_indices)
                img_locs = K.gather(img_locs, top_k_indices)
            # shapeとdtypeを合わせてconcat
            output_size = K.shape(img_classes)[0]
            img_classes = K.reshape(img_classes, (output_size, 1))
            img_confs = K.reshape(img_confs, (output_size, 1))
            img_classes = K.cast(img_classes, K.floatx())
            output = K.concatenate([img_classes, img_confs, img_locs], axis=-1)
            # top_k個に満たなければzero padding。
            output = tf.pad(output, [[0, self.top_k - output_size], [0, 0]])
            return output

    return NMS
