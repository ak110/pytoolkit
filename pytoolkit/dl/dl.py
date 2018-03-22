"""DeepLearning(主にKeras)関連。"""
import copy
import csv
import os
import pathlib
import time
import warnings

import numpy as np
import pandas as pd

from .. import log, utils


def device(cpu=False, gpu=False):
    """TensorFlowのデバイス指定の簡単なラッパー。"""
    assert cpu != gpu
    import tensorflow as tf
    if cpu:
        return tf.device('/cpu:0')
    else:
        return tf.device('/gpu:0')


@log.trace()
def create_data_parallel_model(model, batch_size, gpu_count=None):
    """複数GPUでデータ並列するモデルを作成する。

    # 参考
    https://github.com/fchollet/keras/issues/2436
    https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    """
    if gpu_count is None:
        gpu_count = utils.get_gpu_count()
    if gpu_count <= 1:
        return model, batch_size

    assert isinstance(model.inputs, list)
    assert isinstance(model.outputs, list)

    import keras

    parallel_model = keras.utils.multi_gpu_model(model, gpu_count)

    # Model.saveの置き換え
    # https://github.com/fchollet/keras/issues/2436#issuecomment-294243024
    def _save(self_, *args, **kargs):
        assert self_ is not None  # noqa
        model.save(*args, **kargs)

    def _save_weights(self_, *args, **kargs):
        assert self_ is not None  # noqa
        model.save_weights(*args, **kargs)

    parallel_model.save = type(model.save)(_save, parallel_model)
    parallel_model.save_weights = type(model.save_weights)(_save_weights, parallel_model)

    return parallel_model, batch_size * gpu_count


class Builder(object):
    """Kerasでネットワークを作るときのヘルパークラス。"""

    def __init__(self):
        self.conv_defaults = {'padding': 'same'}
        self.dense_defaults = {}
        self.bn_defaults = {}
        self.act_defaults = {'activation': 'elu'}

    def set_default_l2(self, l2_weight=1e-5):
        """全layerの既定値にL2を設定。"""
        from keras.regularizers import l2
        reg = l2(l2_weight)
        self.conv_defaults['kernel_regularizer'] = reg
        self.conv_defaults['bias_regularizer'] = reg
        self.dense_defaults['kernel_regularizer'] = reg
        self.dense_defaults['bias_regularizer'] = reg
        self.bn_defaults['gamma_regularizer'] = reg
        self.bn_defaults['beta_regularizer'] = reg

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

    def _conv(self, conv, filters, kernel_size, name, use_bn, use_act, bn_kwargs, act_kwargs, **kwargs):
        """ConvND+BN+Act。"""
        kwargs = copy.copy(kwargs)
        bn_kwargs = copy.copy(bn_kwargs) if bn_kwargs is not None else {}
        act_kwargs = copy.copy(act_kwargs) if act_kwargs is not None else {}
        if use_bn:
            kwargs['use_bias'] = False
        if 'activation' in kwargs:
            if use_bn:
                assert 'activation' not in act_kwargs
                act_kwargs['activation'] = kwargs.pop('activation')
            else:
                use_act = False

        if not use_bn:
            assert len(bn_kwargs) == 0
        if not use_act:
            assert len(act_kwargs) == 0
        if not kwargs.get('use_bias', True):
            assert 'bias_initializer' not in kwargs
            assert 'bias_regularizer' not in kwargs
            assert 'bias_constraint' not in kwargs

        conv = conv(filters, kernel_size, name=name, **self._params(self.conv_defaults, kwargs))
        bn = self.bn(name=name + '_bn', **bn_kwargs) if use_bn else None
        act = self.act(name=name + '_act', **act_kwargs) if use_act else None

        if not use_bn and not use_act:
            return conv

        def _pseudo_layer(x):
            x = conv(x)
            x = bn(x) if use_bn else x
            x = act(x) if use_act else x
            return x

        return _pseudo_layer

    def bn(self, **kwargs):
        """BatchNormalization。"""
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


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    return {
        'Destandarization': destandarization_layer(),
        'StocasticAdd': stocastic_add_layer(),
        'NormalNoise': normal_noise_layer(),
        'L2Normalization': l2normalization_layer(),
        'WeightedMean': weighted_mean_layer(),
        'NSGD': nsgd(),
    }


def destandarization_layer():
    """クラスを作って返す。"""
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
            self.supports_masking = True
            self.mean = K.cast_to_floatx(mean)
            self.std = K.cast_to_floatx(std)
            if self.std <= K.epsilon():
                self.std = 1.  # 怪しい安全装置
            super().__init__(**kwargs)

        def call(self, inputs, **kwargs):
            return inputs * self.std + self.mean

        def get_config(self):
            config = {'mean': float(self.mean), 'std': float(self.std)}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return Destandarization


def stocastic_add_layer():
    """クラスを作って返す。"""
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


def normal_noise_layer():
    """クラスを作って返す。"""
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
            config = {
                'noise_rate': self.noise_rate,
            }
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NormalNoise


def l2normalization_layer():
    """クラスを作って返す。"""
    import keras
    import keras.backend as K

    class L2Normalization(keras.engine.topology.Layer):
        """L2 Normalizationレイヤー"""

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


def weighted_mean_layer():
    """クラスを作って返す。"""
    import keras
    import keras.backend as K

    class WeightedMean(keras.engine.topology.Layer):
        """入力の加重平均を取るレイヤー。"""

        def __init__(self,
                     kernel_initializer=keras.initializers.constant(0.1),
                     kernel_regularizer=None,
                     kernel_constraint='non_neg',
                     **kwargs):
            self.supports_masking = True
            self.kernel = None
            self.kernel_initializer = keras.initializers.get(kernel_initializer)
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
            self.kernel_constraint = keras.constraints.get(kernel_constraint)
            super().__init__(**kwargs)

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


def nsgd():
    """重み別に学習率の係数を設定できるSGD+Nesterov momentum Optimizer。"""
    import keras
    import keras.backend as K

    class NSGD(keras.optimizers.SGD):
        """重み別に学習率の係数を設定できるSGD+Nesterov momentum Optimizer。

        lr_multipliersは、Layer.trainable_weights[i]をキーとし、学習率の係数を値としたdict。

        # 例

        ```py
        lr_multipliers = {}
        for layer in basenet.layers:
            w = layer.trainable_weights
            lr_multipliers.update(zip(w, [0.1] * len(w)))
        ```

        """

        def __init__(self, lr=0.1, lr_multipliers=None, momentum=0.9, decay=0., nesterov=True, **kwargs):
            super().__init__(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs)
            self.lr_multipliers = {t if isinstance(t, str) else t.name: mp for t, mp in (lr_multipliers or {}).items()}

        @keras.legacy.interfaces.legacy_get_updates_support
        def get_updates(self, loss, params):
            grads = self.get_gradients(loss, params)
            self.updates = []

            lr = self.lr
            if self.initial_decay > 0:
                lr *= (1. / (1. + self.decay * self.iterations))
                self.updates.append(K.update_add(self.iterations, 1))

            # momentum
            shapes = [K.get_variable_shape(p) for p in params]
            moments = [K.zeros(shape) for shape in shapes]
            self.weights = [self.iterations] + moments
            lr_multipliers = copy.deepcopy(self.lr_multipliers)
            for p, g, m in zip(params, grads, moments):
                mlr = lr * lr_multipliers.pop(p.name) if p.name in lr_multipliers else lr
                v = self.momentum * m - mlr * g  # velocity
                self.updates.append(K.update(m, v))

                if self.nesterov:
                    new_p = p + self.momentum * v - mlr * g
                else:
                    new_p = p + v

                # Apply constraints.
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)

                self.updates.append(K.update(p, new_p))

            assert len(lr_multipliers) == 0, f'Invalid lr_multipliers: {lr_multipliers}'
            return self.updates

        def get_config(self):
            config = {'lr_multipliers': self.lr_multipliers}
            base_config = super().get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NSGD


def session(config=None, gpu_options=None):
    """TensorFlowのセッションの初期化・後始末。

    # 使い方

    ```
    with tk.dl.session():

        # kerasの処理

    ```

    """
    import keras.backend as K

    class _Scope(object):  # pylint: disable=R0903

        def __init__(self, config=None, gpu_options=None):
            self.config = config or {}
            self.gpu_options = gpu_options or {}

        def __enter__(self):
            if K.backend() == 'tensorflow':
                import tensorflow as tf
                self.config.update({'allow_soft_placement': True})
                self.gpu_options.update({'allow_growth': True})
                if 'OMP_NUM_THREADS' in os.environ and 'intra_op_parallelism_threads' not in self.config:
                    self.config['intra_op_parallelism_threads'] = int(os.environ['OMP_NUM_THREADS'])
                config = tf.ConfigProto(**self.config)
                for k, v in self.gpu_options.items():
                    setattr(config.gpu_options, k, v)
                K.set_session(tf.Session(config=config))

        def __exit__(self, *exc_info):
            if K.backend() == 'tensorflow':
                K.clear_session()

    return _Scope(config=config, gpu_options=gpu_options)


def learning_rate_callback(reduce_epoch_rates=(0.5, 0.75), factor=0.1, logger_name=None):
    """よくある150epoch目と225epoch目に学習率を1/10するコールバックを作って返す。"""
    import keras
    import keras.backend as K

    class _LearningRate(keras.callbacks.Callback):

        def __init__(self, reduce_epoch_rates, factor, logger_name):
            self.reduce_epoch_rates = reduce_epoch_rates
            self.factor = factor
            self.logger_name = logger_name
            self.reduce_epochs = None
            super().__init__()

        def on_train_begin(self, logs=None):
            epochs = self.params['epochs']
            self.reduce_epochs = [min(max(int(epochs * r), 1), epochs) for r in self.reduce_epoch_rates]

        def on_epoch_begin(self, epoch, logs=None):
            if epoch + 1 in self.reduce_epochs:
                lr1 = K.get_value(self.model.optimizer.lr)
                lr2 = lr1 * self.factor
                K.set_value(self.model.optimizer.lr, lr2)
                logger = log.get(self.logger_name or __name__)
                logger.info(f'Learning rate: {lr1:.1e} -> {lr2:.1e}')

    return _LearningRate(reduce_epoch_rates=reduce_epoch_rates, factor=factor, logger_name=logger_name)


def learning_curve_plot_callback(filename, metric='loss'):
    """Learning Curvesの描画を行う。

    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - metric: 対象とするmetric名。lossとかaccとか。

    # 「Invalid DISPLAY variable」対策
    最初の方に以下のコードを記述する。
    ```
    import matplotlib as mpl
    mpl.use('Agg')
    ```
    """
    import keras

    class _LearningCurvePlotter(keras.callbacks.Callback):

        def __init__(self, filename, metric='loss'):
            self.filename = pathlib.Path(filename)
            self.metric = metric
            self.met_list = []
            self.val_met_list = []
            super().__init__()

        def on_epoch_end(self, epoch, logs=None):
            try:
                self._plot(logs)
            except BaseException:
                import traceback
                warnings.warn(traceback.format_exc(), RuntimeWarning)

        def _plot(self, logs):
            met = logs.get(self.metric)
            if met is None:
                warnings.warn(f'LearningCurvePlotter requires {self.metric} available!', RuntimeWarning)
            val_met = logs.get(f'val_{self.metric}')

            self.met_list.append(met)
            self.val_met_list.append(val_met)

            if len(self.met_list) > 1:
                df = pd.DataFrame()
                df[self.metric] = self.met_list
                if val_met is not None:
                    df[f'val_{self.metric}'] = self.val_met_list

                df.plot()

                import matplotlib.pyplot as plt
                self.filename.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(self.filename).format(metric=self.metric))
                plt.close()

    return _LearningCurvePlotter(filename=filename, metric=metric)


def tsv_log_callback(filename, append=False):
    """ログを保存するコールバック。

    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - append: 追記するのか否か。

    """
    import keras
    import keras.backend as K

    class _TSVLogger(keras.callbacks.Callback):

        def __init__(self, filename, append):
            self.filename = pathlib.Path(filename)
            self.append = append
            self.log_file = None
            self.log_writer = None
            self.epoch_start_time = None
            super().__init__()

        def on_train_begin(self, logs=None):
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = self.filename.open('a' if self.append else 'w', buffering=65536)
            self.log_writer = csv.writer(self.log_file, delimiter='\t', lineterminator='\n')
            self.log_writer.writerow(['epoch', 'lr'] + self.params['metrics'] + ['time'])

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['lr'] = K.get_value(self.model.optimizer.lr)
            elapsed_time = time.time() - self.epoch_start_time

            def _format_metric(logs, k):
                value = logs.get(k)
                if value is None:
                    return '<none>'
                return f'{value:.4f}'

            metrics = [_format_metric(logs, k) for k in self.params['metrics']]
            self.log_writer.writerow([epoch + 1, '{:.1e}'.format(logs['lr'])] + metrics +
                                     [str(int(np.ceil(elapsed_time)))])
            self.log_file.flush()

        def on_train_end(self, logs=None):
            self.log_file.close()

    return _TSVLogger(filename=filename, append=append)


def logger_callback(name='__main__'):
    """`logging.getLogger(name)`にDEBUGログを色々出力するcallback"""
    import keras
    import keras.backend as K

    class _Logger(keras.callbacks.Callback):

        def __init__(self, name):
            self.name = name
            self.logger = log.get(name)
            self.epoch_start_time = None
            super().__init__()

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            lr = K.get_value(self.model.optimizer.lr)
            elapsed_time = time.time() - self.epoch_start_time
            metrics = " ".join([f'{k}={logs.get(k):.4f}' for k in self.params['metrics']])
            self.logger.debug('Epoch %3d: lr=%.1e %s time=%d',
                              epoch + 1, lr, metrics, int(np.ceil(elapsed_time)))

    return _Logger(name=name)


def freeze_bn_callback(freeze_epoch_rate: float, logger_name=None):
    """指定epoch目でBNを全部freezeする。

    SENetの論文の最後の方にしれっと書いてあったので真似てみた。

    ■Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507

    ## 引数
    - freeze_epoch_rate: 発動するepoch数の割合を指定。
    - logger_name: ログ出力先

    ## 使用例

    ```
    callbacks.append(tk.dl.freeze_bn_callback(0.95))
    ```

    """
    import keras

    class _FreezeBNCallback(keras.callbacks.Callback):

        def __init__(self, freeze_epoch_rate, logger_name):
            self.freeze_epoch_rate = freeze_epoch_rate
            self.logger_name = logger_name
            self.freeze_epoch = 0
            self.freezed_layers = []
            super().__init__()

        def on_train_begin(self, logs=None):
            assert 0 < self.freeze_epoch_rate <= 1
            self.freeze_epoch = int(self.params['epochs'] * self.freeze_epoch_rate)

        def on_epoch_begin(self, epoch, logs=None):
            if self.freeze_epoch == epoch + 1:
                self._freeze_layers(self.model)
                if len(self.freezed_layers) > 0:
                    self._recompile()
                logger = log.get(self.logger_name or __name__)
                logger.info(f'Freeze BN: freezed layers = {len(self.freezed_layers)}')

        def _freeze_layers(self, container):
            for layer in container.layers:
                if isinstance(layer, keras.layers.BatchNormalization):
                    if layer.trainable:
                        layer.trainable = False
                        self.freezed_layers.append(layer)
                elif hasattr(layer, 'layers'):
                    self._freeze_layers(layer)

        def on_train_end(self, logs=None):
            unfreezed = 0
            for layer in self.freezed_layers:
                if not layer.trainable:
                    layer.trainable = True
                    unfreezed += 1
            self.freezed_layers = []
            if unfreezed > 0:
                self._recompile()
            logger = log.get(self.logger_name or __name__)
            logger.info(f'Freeze BN: unfreezed layers = {unfreezed}')

        def _recompile(self):
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics,
                loss_weights=self.model.loss_weights,
                sample_weight_mode=self.model.sample_weight_mode,
                weighted_metrics=self.model.weighted_metrics)

    return _FreezeBNCallback(freeze_epoch_rate=freeze_epoch_rate, logger_name=logger_name)


@log.trace()
def plot_model_params(model, to_file='model.params.png', skip_bn=True):
    """パラメータ数を棒グラフ化"""
    import keras
    import keras.backend as K
    rows = []
    for layer in model.layers:
        if skip_bn and isinstance(layer, keras.layers.BatchNormalization):
            continue
        pc = sum([K.count_params(p) for p in layer.trainable_weights])
        if pc <= 0:
            continue
        rows.append([layer.name, pc])

    df = pd.DataFrame(data=rows, columns=['name', 'params'])
    df.plot(x='name', y='params', kind='barh', figsize=(16, 4 * (len(rows) // 32 + 1)))

    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.savefig(str(to_file))
    plt.close()


def count_trainable_params(model):
    """modelのtrainable paramsを数える"""
    import keras.backend as K
    return sum([sum([K.count_params(p) for p in layer.trainable_weights]) for layer in model.layers])


def count_network_depth(model):
    """重みを持っている層の数を数える。

    「kernel」を持っているレイヤーを数える。
    ConvやDenseなど。ResNet界隈(?)ではDenseは含めないのでずれてしまうが…。
    """
    count = 0
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            count += 1
        elif hasattr(layer, 'layers'):
            count += count_network_depth(layer)
    return count


def categorical_crossentropy(y_true, y_pred, alpha=None):
    """αによるclass=0とそれ以外の重み可変ありのcategorical_crossentropy。"""
    import keras.backend as K
    assert K.image_data_format() == 'channels_last'

    if alpha is None:
        class_weights = -1  # 「-K.sum()」するとpylintが誤検知するのでここに入れ込んじゃう
    else:
        nb_classes = K.int_shape(y_pred)[-1]
        class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
        class_weights = np.reshape(class_weights, (1, 1, -1))
        class_weights = -class_weights  # 「-K.sum()」するとpylintが誤検知するのでここに入れ込んじゃう

    y_pred = K.maximum(y_pred, K.epsilon())
    return K.sum(y_true * K.log(y_pred) * class_weights, axis=-1)


def categorical_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """多クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    import keras.backend as K

    assert K.image_data_format() == 'channels_last'
    nb_classes = K.int_shape(y_pred)[-1]
    class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
    class_weights = np.reshape(class_weights, (1, 1, -1))
    class_weights = -class_weights  # 「-K.sum()」するとpylintが誤検知するのでここに入れ込んじゃう

    y_pred = K.maximum(y_pred, K.epsilon())
    return K.sum(K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) * class_weights, axis=-1)


def od_bias_initializer(nb_classes, pi=0.01):
    """Object Detectionの最後のクラス分類のbias_initializer。nb_classesは背景を含むクラス数。0が背景。"""
    import keras

    class FocalLossBiasInitializer(keras.initializers.Initializer):
        """focal loss用の最後のクラス分類のbias_initializer。

        # 引数
        - nb_classes: 背景を含むクラス数。class 0が背景。
        """

        def __init__(self, nb_classes=None, pi=0.01):
            self.nb_classes = nb_classes
            self.pi = pi

        def __call__(self, shape, dtype=None):
            assert len(shape) == 1
            assert shape[0] % self.nb_classes == 0
            x = np.log(((nb_classes - 1) * (1 - self.pi)) / self.pi)
            bias = [x] + [0] * (nb_classes - 1)  # 背景が0.99%になるような値。21クラス分類なら7.6くらい。(結構大きい…)
            bias = bias * (shape[0] // self.nb_classes)
            import keras.backend as K
            return K.constant(bias, shape=shape, dtype=dtype)

        def get_config(self):
            return {'nb_classes': self.nb_classes}

    return FocalLossBiasInitializer(nb_classes, pi)


def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    import keras.backend as K
    import tensorflow as tf
    abs_loss = K.abs(y_true - y_pred)
    sq_loss = 0.5 * K.square(y_true - y_pred)
    l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    l1_loss = K.sum(l1_loss, axis=-1)
    return l1_loss


@log.trace()
def load_weights(model, filepath, where_fn=None):
    """重みの読み込み。

    model.load_weights()は重みの形が違うと読み込めないが、
    警告を出しつつ読むようにしたもの。

    # 引数
    - model: 読み込み先モデル。
    - filepath: モデルのファイルパス。(str or pathlib.Path)
    - where_fn: 読み込むレイヤー名を受け取り、読み込むか否かを返すcallable。
    """
    import h5py
    import keras
    import keras.backend as K
    with h5py.File(str(filepath), mode='r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        original_keras_version = f.attrs['keras_version'].decode('utf8') if 'keras_version' in f.attrs else '1'
        original_backend = f.attrs['backend'].decode('utf8') if 'backend' in f.attrs else None

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]  # noqa

        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            if where_fn is not None and not where_fn(name):
                continue

            try:
                layer = model.get_layer(name=name)
            except ValueError as e:
                warnings.warn(str(e))
                continue

            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]

            symbolic_weights = layer.weights
            weight_values = keras.engine.topology.preprocess_weights_for_loading(
                layer,
                weight_values,
                original_keras_version,
                original_backend)
            if len(weight_values) != len(symbolic_weights):
                warnings.warn('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' +
                              str(len(symbolic_weights)) + ' weight(s), but the saved weights' +
                              ' have ' + str(len(weight_values)) + ' element(s).')
                continue
            is_match_shapes = True
            for s, w in zip(symbolic_weights, weight_values):
                if s.shape != w.shape:
                    warnings.warn('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' +
                                  str(s.shape) + ' weight(s), but the saved weights' +
                                  ' have ' + str(w.shape) + ' element(s).')
                    is_match_shapes = False
                    continue
            if is_match_shapes:
                for s, w in zip(symbolic_weights, weight_values):
                    weight_value_tuples.append((s, w))
        K.batch_set_value(weight_value_tuples)
