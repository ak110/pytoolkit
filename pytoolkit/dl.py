"""DeepLearning(主にKeras)関連。

kerasをimportしてしまうとTensorFlowの初期化が始まって重いので、
importしただけではkerasがimportされないように作っている。

"""
import contextlib
import copy
import csv
import os
import pathlib
import queue
import threading
import time
import warnings

import numpy as np
import pandas as pd
import sklearn.utils

from . import log, utils


def device(cpu=False, gpu=False):
    """`tf.device('/cpu:0')` などの簡単なラッパー。"""
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

    def set_default_l2(self, l=1e-5):
        """全layerの既定値にL2を設定。"""
        from keras.regularizers import l2
        reg = l2(l)
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


def conv2d(filters, kernel_size, activation, name, use_bn=True, use_batch_renorm=False, preact=False, **kargs):
    """Conv2D+BN+Activationの簡単なヘルパー。"""
    import keras
    if use_bn:
        def _conv2d(x):
            if not preact:
                x = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name=name, **kargs)(x)
            if use_batch_renorm:
                import keras_contrib
                x = keras_contrib.layers.BatchRenormalization(name=name + '_brn')(x)
            else:
                x = keras.layers.BatchNormalization(name=name + '_bn')(x)
            if activation is not None:
                x = keras.layers.Activation(activation, name=name + '_act')(x)
            if preact:
                x = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name=name, **kargs)(x)
            return x
        return _conv2d
    else:
        if preact:
            def _conv2d(x):
                if activation is not None:
                    x = keras.layers.Activation(activation, name=name + '_act')(x)
                x = keras.layers.Conv2D(filters, kernel_size, name=name, **kargs)(x)
                return x
            return _conv2d
        else:
            return keras.layers.Conv2D(filters, kernel_size, activation=activation, name=name, **kargs)


def sepconv2d(filters, kernel_size, activation, name, use_bn=True, use_batch_renorm=False, preact=False, **kargs):
    """SeparableConv2D+BN+Activationの簡単なヘルパー。"""
    import keras
    if use_bn:
        def _conv2d(x):
            if not preact:
                x = keras.layers.SeparableConv2D(filters, kernel_size, use_bias=False, name=name, **kargs)(x)
            if use_batch_renorm:
                import keras_contrib
                x = keras_contrib.layers.BatchRenormalization(name=name + '_brn')(x)
            else:
                x = keras.layers.BatchNormalization(name=name + '_bn')(x)
            x = keras.layers.Activation(activation, name=name + '_act')(x)
            if preact:
                x = keras.layers.SeparableConv2D(filters, kernel_size, use_bias=False, name=name, **kargs)(x)
            return x
        return _conv2d
    else:
        if preact:
            def _conv2d(x):
                x = keras.layers.Activation(activation, name=name + '_act')(x)
                x = keras.layers.SeparableConv2D(filters, kernel_size, name=name, **kargs)(x)
                return x
            return _conv2d
        else:
            return keras.layers.SeparableConv2D(filters, kernel_size, activation=activation, name=name, **kargs)


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    return {
        'Destandarization': destandarization_layer(),
        'StocasticAdd': stocastic_add_layer(),
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
            self.lr_multipliers = {t if isinstance(t, str) else t.name: mp for t, mp in lr_multipliers.items()}

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

            assert len(lr_multipliers) == 0, 'Invalid lr_multipliers: {}'.format(lr_multipliers)
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


def learning_rate_callback(lr=0.1, epochs=300):
    """よくある150epoch目と225epoch目に学習率を1/10するコールバックを作って返す。"""
    assert epochs % 4 == 0
    import keras
    lr_list = [lr] * (epochs // 2) + [lr / 10] * (epochs // 4) + [lr / 100] * (epochs // 4)
    return keras.callbacks.LearningRateScheduler(lambda ep: lr_list[ep])


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
            except:  # noqa
                import traceback
                warnings.warn(traceback.format_exc(), RuntimeWarning)

        def _plot(self, logs):
            met = logs.get(self.metric)
            if met is None:
                warnings.warn('LearningCurvePlotter requires {} available!'.format(self.metric), RuntimeWarning)
            val_met = logs.get('val_{}'.format(self.metric))

            self.met_list.append(met)
            self.val_met_list.append(val_met)

            if len(self.met_list) > 1:
                df = pd.DataFrame()
                df[self.metric] = self.met_list
                if val_met is not None:
                    df['val_{}'.format(self.metric)] = self.val_met_list

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
            metrics = ['{:.4f}'.format(logs.get(k)) for k in self.params['metrics']]
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
            metrics = " ".join(['{}={:.4f}'.format(k, logs.get(k)) for k in self.params['metrics']])
            self.logger.debug('Epoch %3d: lr=%.1e %s time=%d',
                              epoch + 1, lr, metrics, int(np.ceil(elapsed_time)))

    return _Logger(name=name)


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


class Generator(object):
    """`fit_generator`などに渡すgeneratorを作るためのベースクラス。

    # 引数
    - data_encoder: Xの変換を行う関数
    - label_encoder: yの変換を行う関数
    - parallel: _prepareにjoblib.Parallelを渡すならTrue

    """

    def __init__(self, data_encoder=None, label_encoder=None):
        self.data_encoder = data_encoder
        self.label_encoder = label_encoder

    @staticmethod
    def steps_per_epoch(data_count, batch_size):
        """1epochが何ステップかを算出して返す"""
        return (data_count + batch_size - 1) // batch_size

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, data_augmentation=False, random_state=None):
        """`fit_generator`などに渡すgenerator。kargsはそのままprepareに渡される。"""
        random_state = sklearn.utils.check_random_state(random_state)
        length = len(X[0]) if isinstance(X, list) else len(X)
        if y is not None:
            assert length == (len(y[0]) if isinstance(y, list) else len(y))

        cpu_count = os.cpu_count()
        max_queue_size = max(batch_size * 4, cpu_count * 2)  # 適当に余裕をもったサイズにしておく
        worker_count = min(batch_size * 4, cpu_count * 2)
        input_queue = queue.Queue(maxsize=max_queue_size)
        result_queue = queue.Queue(maxsize=max_queue_size)
        workers = [threading.Thread(target=self._worker,
                                    args=(wid, input_queue, result_queue, data_augmentation),
                                    daemon=True)
                   for wid in range(worker_count)]
        for worker in workers:
            worker.start()
        try:
            gen = self._flow_index(length, shuffle, random_state)
            result_buffer = []
            next_seq = 0
            seq_in = 0

            while True:
                # キューサイズの限界までinput_queueにデータを入れる
                queue_size = input_queue.qsize() + result_queue.qsize()
                while queue_size < max_queue_size:
                    ix, seed, x_, y_, w_ = self._pick_next(gen, X, y, weights)
                    input_queue.put((seq_in, ix, seed, x_, y_, w_))
                    queue_size += 1
                    seq_in += 1

                # バッチサイズ分の結果を取り出す
                remain_size = length - next_seq % length
                cur_batch_size = batch_size if shuffle or batch_size <= remain_size else remain_size
                while True:
                    if len(result_buffer) >= cur_batch_size:
                        if shuffle:
                            break
                        else:
                            result_buffer.sort(key=lambda x: x[0])
                            first_seq = result_buffer[0][0]
                            last_seq = result_buffer[cur_batch_size - 1][0]
                            if first_seq == next_seq and last_seq == next_seq + cur_batch_size - 1:
                                break
                    result_buffer.append(result_queue.get())
                _, rx, ry, rw = zip(*result_buffer[:cur_batch_size])
                result_buffer = result_buffer[cur_batch_size:]
                next_seq += cur_batch_size

                # 結果を返す
                yield self._get_result(X, y, weights, rx, ry, rw)
        except GeneratorExit:
            pass
        finally:
            # 処理中のものをいったんキャンセルして、
            # Noneを入れて、結果キューを空にして、join
            with contextlib.suppress(queue.Empty):
                while True:
                    input_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                while True:
                    input_queue.put_nowait((None, None, None, None, None, None))
            with contextlib.suppress(queue.Empty):
                while True:
                    result_queue.get_nowait()
            for worker in workers:
                worker.join()

    @staticmethod
    def _pick_next(gen, X, y, weights):
        """genから1件取り出す。"""
        def _pick(arr, ix):
            return [x[ix] for x in arr] if isinstance(arr, list) else arr[ix]

        ix, seed = next(gen)
        x_ = _pick(X, ix)
        if y is None:
            y_ = None
        else:
            y_ = _pick(y, ix)
        if weights is None:
            w_ = None
        else:
            w_ = weights[ix]
        return ix, seed, x_, y_, w_

    @staticmethod
    def _get_result(X, y, weights, rx, ry, rw):
        """Kerasに渡すデータを返す。"""
        def _arr(arr, islist):
            return [np.array(a) for a in arr] if islist else np.array(arr)

        if y is None:
            assert weights is None
            return _arr(rx, isinstance(X, list))
        elif weights is None:
            return _arr(rx, isinstance(X, list)), _arr(ry, isinstance(y, list))
        else:
            return _arr(rx, isinstance(X, list)), _arr(ry, isinstance(y, list)), np.array(rw)

    @staticmethod
    def _flow_index(data_count, shuffle, random_state):
        """データのindexとseedを列挙し続けるgenerator。"""
        indices = np.arange(data_count)
        while True:
            if shuffle:
                random_state.shuffle(indices)
            seeds = random_state.randint(0, 2 ** 31, size=(len(indices),))
            for index, seed in zip(indices, seeds):
                yield index, seed

    def _worker(self, wid, input_queue, result_queue, data_augmentation):
        """ワーカースレッド。"""
        assert wid >= 0
        while True:
            seq, ix, seed, x_, y_, w_ = input_queue.get()
            if seq is None:
                break
            x_, y_, w_ = self.generate(ix, seed, x_, y_, w_, data_augmentation)
            result_queue.put((seq, x_, y_, w_))
        # allow_exit_without_flush
        # result_queue.cancel_join_thread()

    def generate(self, ix, seed, x_, y_, w_, data_augmentation):
        """1件分の処理。

        画像の読み込みとかDataAugmentationとか。
        y_やw_は使わない場合もそのまま返せばOK。(使う場合はNoneに注意。)
        """
        assert ix is not None
        assert seed is not None
        assert data_augmentation in (True, False)
        if self.data_encoder:
            x_ = np.squeeze(self.data_encoder(np.expand_dims(x_, 0)), axis=0)
        if self.label_encoder and y_ is not None:
            y_ = np.squeeze(self.label_encoder(np.expand_dims(y_, 0)), axis=0)
        return x_, y_, w_


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
