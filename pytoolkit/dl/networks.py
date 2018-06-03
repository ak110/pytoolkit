"""Kerasでネットワークを作るときのヘルパーなど。"""
import copy

from . import layers


class Builder(object):
    """Kerasでネットワークを作るときのヘルパークラス。"""

    def __init__(self, default_l2=1e-5):
        import keras
        self.conv_defaults = {'kernel_initializer': 'he_uniform', 'padding': 'same', 'use_bias': False}
        self.dense_defaults = {'kernel_initializer': 'he_uniform'}
        self.bn_defaults = {}
        self.act_defaults = {'activation': 'elu'}
        self.bn_class = keras.layers.BatchNormalization
        self.act_class = keras.layers.Activation
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

    def conv1d(self, filters, kernel_size=3, name=None, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv1D+BN+Act。"""
        import keras.layers
        return self._conv(1, keras.layers.Conv1D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def conv2d(self, filters, kernel_size=3, name=None, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv2D+BN+Act。"""
        import keras.layers
        return self._conv(2, keras.layers.Conv2D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def conv3d(self, filters, kernel_size=3, name=None, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv3D+BN+Act。"""
        import keras.layers
        return self._conv(3, keras.layers.Conv3D, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def conv2dtr(self, filters, kernel_size=3, name=None, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """Conv2DTranspose+BN+Act。"""
        import keras.layers
        return self._conv(2, keras.layers.Conv2DTranspose, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def dwconv2d(self, kernel_size=3, name=None, use_bn=True, use_act=True, bn_kwargs=None, act_kwargs=None, **kwargs):
        """DepthwiseConv2D+BN+Act。"""
        import keras.layers
        return self._conv(2, keras.layers.DepthwiseConv2D, None, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs)

    def _conv(self, rank, conv, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs):
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

        seq = []
        if kwargs['padding'] in ('reflect', 'symmetric'):
            kernel_size = keras.utils.conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
            strides = keras.utils.conv_utils.normalize_tuple(kwargs.get('strides', 1), rank, 'strides')
            if kernel_size == strides:
                pass  # padding無しのはず
            elif len(kernel_size) == 2:
                pad = (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
                seq.append(layers.pad2d()(pad, mode=kwargs['padding'], name=f'{name}_pad' if name is not None else None))
            else:
                assert False  # 未対応
            kwargs['padding'] = 'valid'
        seq.append(conv(*args, **kwargs))
        if use_bn:
            seq.append(self.bn(name=f'{name}_bn' if name is not None else None, **bn_kwargs))
        if use_act:
            seq.append(self.act(name=f'{name}_act' if name is not None else None, **act_kwargs))
        return Sequence(seq)

    def bn_act(self, name=None):
        """BN+Act。"""
        bn_name = None if name is None else name + '_bn'
        act_name = None if name is None else name + '_act'
        return Sequence([self.bn(name=bn_name), self.act(name=act_name)])

    def bn(self, **kwargs):
        """BatchNormalization。"""
        return self.bn_class(**self._params(self.bn_defaults, kwargs))

    def act(self, **kwargs):
        """Activation。"""
        return self.act_class(**self._params(self.act_defaults, kwargs))

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

    def res_block(self, filters, dropout=None, name=None):
        """普通のResidual Block。((3, 3) × 2)"""
        import keras
        seq = []
        seq.append(self.conv2d(filters, use_act=True, name=f'{name}_conv1' if name else None))
        if dropout:
            seq.append(keras.layers.Dropout(dropout, name=f'{name}_drop' if name else None))
        seq.append(self.conv2d(filters, use_act=False, name=f'{name}_conv2' if name else None))
        return Sequence(seq, keras.layers.Add(name=f'{name}_add' if name else None))

    def se_block(self, filters, ratio=16, name=None):
        """Squeeze-and-Excitation block

        https://arxiv.org/abs/1709.01507
        """
        import keras
        reg = keras.regularizers.l2(1e-4)
        seq = [
            keras.layers.GlobalAveragePooling2D(name=f'{name}_p' if name else None),
            self.dense(filters // ratio, activation='relu', kernel_regularizer=reg,
                       name=f'{name}_sq' if name else None),
            self.dense(filters, activation='sigmoid', kernel_regularizer=reg,
                       name=f'{name}_ex' if name else None),
            keras.layers.Reshape((1, 1, filters), name=f'{name}_r' if name else None),
        ]
        return Sequence(seq, keras.layers.Multiply(name=f'{name}_s' if name else None))


class Sequence(object):
    """複数のレイヤーの塊。kerasのlayer風にcall出来るもの。(プロパティなどは必要に応じて実装予定。。)

    # 引数
    - seq: Kerasのレイヤーの配列。
    - merge_layer: 入力とlayers適用後のをマージする時に使用するレイヤーを指定する。Noneなら何もしない。

    """

    def __init__(self, seq, merge_layer=None):
        self.seq = seq
        self.merge_layer = merge_layer

    def __call__(self, x):
        x_in = x
        for layer in self.seq:
            x = layer(x)
        if self.merge_layer is not None:
            x = self.merge_layer([x_in, x])
        return x

    @property
    def name(self):
        """先頭のレイヤーのnameを返しちゃう。"""
        return self.seq[0].name

    @property
    def names(self):
        """全レイヤーのnameを配列で返す。"""
        return [layer.name for layer in self.seq]

    @property
    def trainable_weights(self):
        """訓練対象の重みTensorを返す。"""
        return sum([layer.trainable_weights for layer in self.seq], [])
