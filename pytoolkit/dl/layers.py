"""Kerasのカスタムレイヤーなど。"""
import copy

import numpy as np


class Builder(object):
    """Kerasでネットワークを作るときのヘルパークラス。"""

    def __init__(self, use_gn=False, default_l2=1e-5):
        self.conv_defaults = {'kernel_initializer': 'he_uniform', 'padding': 'same', 'use_bias': False}
        self.dense_defaults = {'kernel_initializer': 'he_uniform'}
        self.bn_defaults = {}
        self.gn_defaults = {}
        self.act_defaults = {'activation': 'elu'}
        self.use_gn = use_gn
        if default_l2:
            self.set_default_l2(default_l2)

    def set_default_l2(self, default_l2=1e-5):
        """全layerの既定値にL2を設定。"""
        from keras.regularizers import l2
        reg = l2(default_l2)
        self.conv_defaults['kernel_regularizer'] = reg
        self.conv_defaults['bias_regularizer'] = reg
        self.dense_defaults['kernel_regularizer'] = reg
        self.dense_defaults['bias_regularizer'] = reg
        self.bn_defaults['gamma_regularizer'] = reg
        self.bn_defaults['beta_regularizer'] = reg
        self.gn_defaults['gamma_regularizer'] = reg
        self.gn_defaults['beta_regularizer'] = reg

    def conv1d(self, filters, kernel_size, name, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv1D+BN+Act。"""
        import keras.layers
        return self._conv(keras.layers.Conv1D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def conv2d(self, filters, kernel_size, name, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv2D+BN+Act。"""
        import keras.layers
        return self._conv(keras.layers.Conv2D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def conv3d(self, filters, kernel_size, name, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv3D+BN+Act。"""
        import keras.layers
        return self._conv(keras.layers.Conv3D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def conv2dtr(self, filters, kernel_size, name, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv2DTranspose+BN+Act。"""
        import keras.layers
        return self._conv(keras.layers.Conv2DTranspose, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def dwconv2d(self, filters, kernel_size, name, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """DepthwiseConv2D+BN+Act。"""
        import keras.layers
        return self._conv(keras.layers.DepthwiseConv2D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def _conv(self, conv, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs):
        """ConvND+BN+Act。"""
        import keras.layers
        kwargs = copy.copy(kwargs)
        bn_kwargs = copy.copy(bn_kwargs) if bn_kwargs is not None else {}
        act_kwargs = copy.copy(act_kwargs) if act_kwargs is not None else {}
        if not use_bn:
            assert len(bn_kwargs) == 0
        if not use_act:
            assert len(act_kwargs) == 0

        if 'activation' in kwargs:
            if use_bn:
                assert 'activation' not in act_kwargs
                act_kwargs['activation'] = kwargs.pop('activation')
            else:
                use_act = False

        args = [filters, kernel_size]
        kwargs = self._params(self.conv_defaults, kwargs)
        kwargs['name'] = name
        if conv == keras.layers.DepthwiseConv2D:
            kwargs = {k.replace('kernel_', 'depthwise_'): v for k, v in kwargs.items()}
            args = args[1:]

        if not kwargs['use_bias']:
            assert 'bias_initializer' not in kwargs
            assert 'bias_constraint' not in kwargs

        layers = []
        if kwargs['padding'] in ('reflect', 'symmetric'):
            layers.append(pad2d()(mode=kwargs['padding']))  # とりあえず3x3 convのみ対応
            kwargs['padding'] = 'valid'
        layers.append(conv(*args, **kwargs))
        if use_bn:
            layers.append(self.bn(name=name + '_bn', **bn_kwargs))
        if use_act:
            layers.append(self.act(name=name + '_act', **act_kwargs))
        return Sequence(layers)

    def bn_act(self, name=None):
        """BN+Act。"""
        bn_name = None if name is None else name + '_bn'
        act_name = None if name is None else name + '_act'
        return Sequence([self.bn(name=bn_name), self.act(name=act_name)])

    def bn(self, **kwargs):
        """BatchNormalization。"""
        if self.use_gn:
            return group_normalization()(**self._params(self.gn_defaults, kwargs))
        import keras.layers
        return keras.layers.BatchNormalization(**self._params(self.bn_defaults, kwargs))

    def act(self, **kwargs):
        """Activation。"""
        import keras.layers
        return keras.layers.Activation(**self._params(self.act_defaults, kwargs))

    def dense(self, units, **kwargs):
        """Dense。"""
        import keras.layers
        return keras.layers.Dense(units, **self._params(self.dense_defaults, kwargs))

    @staticmethod
    def _params(defaults, kwargs):
        params = copy.copy(defaults)
        params.update(kwargs)
        return params

    @staticmethod
    def shape(x):
        """`K.int_shapeを実行するだけのヘルパー (これだけのためにbackendをimportするのが面倒なので)`"""
        import keras.backend as K
        return K.int_shape(x)

    @staticmethod
    def se_block(filters, ratio=16, name=None):
        """Squeeze-and-Excitation block

        https://arxiv.org/abs/1709.01507
        """
        import keras
        reg = keras.regularizers.l2(1e-4)
        layers = [
            keras.layers.GlobalAveragePooling2D(name=f'{name}_p' if name else None),
            keras.layers.Dense(filters // ratio, activation='relu', use_bias=False,
                               kernel_initializer='he_uniform', kernel_regularizer=reg,
                               name=f'{name}_sq' if name else None),
            keras.layers.Dense(filters, activation='sigmoid', use_bias=False,
                               kernel_initializer='he_uniform', kernel_regularizer=reg,
                               name=f'{name}_ex' if name else None),
            keras.layers.Reshape((1, 1, filters), name=f'{name}_r' if name else None),
        ]
        return Sequence(layers, 'multiply', f'{name}_s' if name else None)


class Sequence(object):
    """複数のレイヤーの塊。kerasのlayer風にcall出来るもの。(プロパティなどは必要に応じて実装予定。。)

    # 引数
    - layers: Kerasのレイヤーの配列。
    - merge_function: callableまたは文字列で、入力とlayers適用後のをマージする関数を指定する。Noneなら何もしなくて'add'ならResidual blockになる。
    - merge_name: merge_functionのname

    """

    def __init__(self, layers, merge_function=None, merge_name=None):
        self.layers = layers
        self.merge_function = merge_function
        self.merge_name = merge_name

    def __call__(self, x):
        x_in = x
        for layer in self.layers:
            x = layer(x)
        if self.merge_function is not None:
            import keras
            merge_fn = self.merge_function if callable(self.merge_function) else keras.layers.__dict__[self.merge_function]
            x = merge_fn([x_in, x], name=self.merge_name)
        return x

    @property
    def name(self):
        """先頭のレイヤーのnameを返しちゃう。"""
        return self.layers[0].name

    @property
    def names(self):
        """全レイヤーのnameを配列で返す。"""
        return [layer.name for layer in self.layers]

    @property
    def trainable_weights(self):
        """訓練対象の重みTensorを返す。"""
        return sum([layer.trainable_weights for layer in self.layers], [])


def channel_argmax():
    """チャンネルをargmaxするレイヤー。"""
    import keras
    import keras.backend as K

    class ChannelArgMax(keras.engine.topology.Layer):
        """チャンネルをargmaxするレイヤー。"""

        def call(self, inputs, **kwargs):
            return K.argmax(inputs, axis=-1)

        def compute_output_shape(self, input_shape):
            return input_shape[:-1]

    return ChannelArgMax


def channel_max():
    """チャンネルをmaxするレイヤー。"""
    import keras
    import keras.backend as K

    class ChannelMax(keras.engine.topology.Layer):
        """チャンネルをmaxするレイヤー。"""

        def call(self, inputs, **kwargs):
            return K.max(inputs, axis=-1)

        def compute_output_shape(self, input_shape):
            return input_shape[:-1]

    return ChannelMax


def pad2d():
    """`tf.pad`するレイヤー。"""
    import keras
    import keras.backend as K
    import tensorflow as tf

    class Pad2D(keras.engine.topology.Layer):
        """`tf.pad`するレイヤー。"""

        def __init__(self, padding=(1, 1), mode='constant', constant_values=0, **kwargs):
            super().__init__(**kwargs)

            assert isinstance(padding, tuple) and len(padding) == 2
            assert mode in ('constant', 'reflect', 'symmetric')

            if isinstance(padding, int):
                padding = ((padding, padding), (padding, padding))
            elif isinstance(padding[0], int):
                padding = ((padding[0], padding[0]), (padding[1], padding[1]))

            self.padding = padding
            self.mode = mode
            self.constant_values = constant_values

        def call(self, inputs, **kwargs):
            padding = K.constant(((0, 0),) + self.padding + ((0, 0),))
            return tf.pad(inputs, padding, self.mode, self.constant_values)

        def compute_output_shape(self, input_shape):
            return input_shape[0] + sum(self.padding[0]), input_shape[1] + sum(self.padding[1]), input_shape[2]

        def get_config(self):
            config = {
                'padding': self.padding,
                'mode': self.mode,
                'constant_values': self.constant_values,
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return Pad2D


def group_normalization():
    """Group normalization。"""
    import keras
    import keras.backend as K
    import tensorflow as tf

    class GroupNormalization(keras.engine.topology.Layer):
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

        def call(self, inputs, **kwargs):
            x = inputs
            shape = K.shape(x)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            x = K.reshape(x, [N, H, W, self.groups, C // self.groups])
            mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
            x = (x - mean) / K.sqrt(var + self.epsilon)
            x = K.reshape(x, [N, H, W, C])
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

    class Destandarization(keras.engine.topology.Layer):
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

    class StocasticAdd(keras.engine.topology.Layer):
        """Stocastic Depthのための確率的な加算。

        # 引数
        - p: survival probability。1だとdropせず、0.5だと1/2の確率でdrop

        """

        def __init__(self, survival_prob, calibration=True, **kargs):
            assert 0 < survival_prob <= 1
            self.survival_prob = float(survival_prob)
            self.calibration = calibration
            super().__init__(**kargs)

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

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) == 2
            assert input_shape[0] == input_shape[1]
            return input_shape[0]

        def get_config(self):
            config = {
                'survival_prob': self.survival_prob,
                'calibration': self.calibration,
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return StocasticAdd


def normal_noise():
    """平均0、分散1のノイズをドロップアウト風に適用する。"""
    import keras
    import keras.backend as K

    class NormalNoise(keras.engine.topology.Layer):
        """平均0、分散1のノイズをドロップアウト風に適用する。"""

        def __init__(self, noise_rate=0.25, **kargs):
            assert 0 <= noise_rate < 1
            self.noise_rate = noise_rate
            super().__init__(**kargs)

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
            config = {'noise_rate': self.noise_rate, }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NormalNoise


def l2normalization():
    """L2 Normalizationレイヤー。"""
    import keras
    import keras.backend as K

    class L2Normalization(keras.engine.topology.Layer):
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
                                         initializer=keras.initializers.constant(init_gamma),
                                         trainable=True)
            return super().build(input_shape)

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

    class WeightedMean(keras.engine.topology.Layer):
        """入力の加重平均を取るレイヤー。"""

        def __init__(self,
                     kernel_initializer=keras.initializers.constant(0.1),
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

        def call(self, inputs, **kwargs):
            ot = K.zeros_like(inputs[0])
            for i, inp in enumerate(inputs):
                ot += inp * self.kernel[i]
            ot /= K.sum(self.kernel) + K.epsilon()
            return ot

        def compute_output_shape(self, input_shape):
            assert input_shape and len(input_shape) >= 2
            return input_shape[0]

        def get_config(self):
            config = {
                'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
                'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return WeightedMean
