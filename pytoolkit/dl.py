"""DeepLearning(主にKeras)関連。

kerasをimportしてしまうとTensorFlowの初期化が始まって重いので、
importしただけではkerasがimportされないように作っている。

"""
import csv
import os
import pathlib
import time
import warnings

import joblib  # pip install joblib
import numpy as np
import pandas as pd
import sklearn.utils

from . import utils


def create_data_parallel_model(model, gpu_count=None):
    """複数GPUでデータ並列するモデルを作成する。

    # 参考
    https://github.com/fchollet/keras/issues/2436
    https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py

    """
    import keras
    import keras.backend as K
    import tensorflow as tf

    if gpu_count is None:
        gpu_count = utils.get_gpu_count()
    if gpu_count <= 1:
        return model

    assert isinstance(model.inputs, list)
    assert isinstance(model.outputs, list)

    # 各GPUでモデルを実行
    tower_outputs = []
    for gpu in range(gpu_count):
        with tf.device('/gpu:%d' % gpu), tf.name_scope('tower_%d' % gpu):  # pylint: disable=E1129
            def _slice(x, index, count):
                input_shape = K.shape(x)
                sliced_batch_size = input_shape[0] // count
                if index == count - 1:
                    return x[sliced_batch_size * index:]
                return x[sliced_batch_size * index:sliced_batch_size * (index + 1)]

            args = {'index': gpu, 'count': gpu_count}
            sliced_inputs = [keras.layers.Lambda(_slice, arguments=args)(x)
                             for x in model.inputs]

            if len(sliced_inputs) == 1:
                sliced_inputs = sliced_inputs[0]
            outputs = model(sliced_inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]

            tower_outputs.append(outputs)

    # 結果をマージ
    with tf.device('/cpu:0'):
        def _concat(inputs):
            return K.concatenate(inputs, axis=0)

        merged = [keras.layers.Lambda(_concat)([to[i] for to in tower_outputs])
                  for i
                  in range(len(model.outputs))]

        parallel_model = keras.models.Model(model.inputs, merged, model.name + '_parallel')

    # Model.saveの置き換え
    # https://github.com/fchollet/keras/issues/2436#issuecomment-294243024
    def _save(self_, *args, **kargs):
        assert self_ is not None  # noqa
        model.save(*args, **kargs)

    parallel_model.save = type(model.save)(_save, parallel_model)

    return parallel_model


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
            x = keras.layers.Activation(activation, name=name + '_act')(x)
            if preact:
                x = keras.layers.Conv2D(filters, kernel_size, use_bias=False, name=name, **kargs)(x)
            return x
        return _conv2d
    else:
        if preact:
            def _conv2d(x):
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
    """独自レイヤーのdictを返す。"""
    return {
        'Destandarization': destandarization_layer(),
        'StocasticAdd': stocastic_add_layer(),
        'L2Normalization': l2normalization_layer(),
        'WeightedMean': weighted_mean_layer(),
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


def my_callback_factory():
    """クラスを作って返す。"""
    import keras
    import keras.backend as K

    class _MyCallback(keras.callbacks.Callback):
        """色々入りのKerasのCallbackクラスを作って返す。

        TerminateOnNaN+ReduceLROnPlateau+EarlyStopping+CSVLoggerのようなもの。

        lossを監視して学習率を制御して、十分学習できたら終了する。
        ついでにログも出す。(ミニバッチ単位＆エポック単位)

        # 引数

        - log_dir: ログ出力先
        - batch_log_name: ミニバッチ単位のログファイル名
        - epoch_log_name: エポック単位のログファイル名

        - lr_list: epoch毎の学習率のリスト (base_lrと排他)

        - base_lr: 学習率の自動調整のベースとする学習率 (lr_listと排他)
        - max_reduces: 学習率の自動調整時、最大で何回学習率を減らすのか
        - reduce_factor: 学習率の自動調整時、学習率を減らしていく割合
        - beta1: lossを監視する際の指数移動平均の係数
        - beta2: lossを監視する際の指数移動平均の係数
        - margin_iterations: 学習率の自動調整時、このバッチ数分までは誤差を考慮して学習率を下げない

        - verbose: 学習率などをprintするなら1

        """

        def __init__(self, log_dir='.',
                     lr_list=None,
                     base_lr=None,
                     verbose=1,
                     batch_log_name='batchlog.tsv',
                     epoch_log_name='epochlog.tsv',
                     append=False,
                     max_reduces=2, reduce_factor=0.1,
                     beta1=0.998, beta2=0.999, margin_iterations=100):
            super().__init__()
            # 設定
            assert (lr_list is None) != (base_lr is None)  # どちらか片方のみ必須
            self.log_dir = log_dir
            self.batch_log_name = batch_log_name
            self.epoch_log_name = epoch_log_name
            self.append = append
            self.lr_list = lr_list
            self.base_lr = base_lr
            self.max_reduces = max_reduces
            self.reduce_factor = reduce_factor
            self.beta1 = beta1
            self.beta2 = beta2
            self.margin_iterations = margin_iterations
            self.verbose = verbose
            # あとで使うものたち
            self.batch_log_file = None
            self.epoch_log_file = None
            self.batch_log_writer = None
            self.epoch_log_writer = None
            self.iterations = 0
            self.iterations_per_reduce = 0
            self.ema1 = 0
            self.ema2 = 0
            self.delta_ema = 0
            self.reduces = 0
            self.stopped_epoch = 0
            self.epoch = 0
            self.epoch_start_time = 0
            self.reduce_on_epoch_end = False

        def on_train_begin(self, logs=None):
            # ログファイル作成
            d = pathlib.Path(self.log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.batch_log_file = d.joinpath(self.batch_log_name).open('a' if self.append else 'w', buffering=65536)
            self.epoch_log_file = d.joinpath(self.epoch_log_name).open('a' if self.append else 'w', buffering=65536)
            self.batch_log_writer = csv.writer(self.batch_log_file, delimiter='\t', lineterminator='\n')
            self.epoch_log_writer = csv.writer(self.epoch_log_file, delimiter='\t', lineterminator='\n')
            self.batch_log_writer.writerow(['epoch', 'batch', 'loss', 'delta_ema'])
            self.epoch_log_writer.writerow(['epoch', 'lr'] + self.params['metrics'] + ['time', 'delta_ema'])
            # 学習率の設定(base_lr)
            if self.base_lr is not None:
                K.set_value(self.model.optimizer.lr, float(self.base_lr))
                if self.verbose >= 1:
                    print('lr = {:.1e}'.format(float(K.get_value(self.model.optimizer.lr))))
            # 色々初期化
            self.iterations = 0
            self.iterations_per_reduce = 0
            self.ema1 = 0
            self.ema2 = 0
            self.reduces = 0
            self.stopped_epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            if self.lr_list is not None:
                # 学習率の設定(lr_list)
                lr = self.lr_list[epoch]
                if self.verbose >= 1:
                    if epoch == 0 or lr != self.lr_list[epoch - 1]:
                        print('lr = {:.1e}'.format(float(lr)))
                K.set_value(self.model.optimizer.lr, float(lr))
            elif self.reduce_on_epoch_end:
                if self.verbose >= 1:
                    print('lr = {:.1e}'.format(float(K.get_value(self.model.optimizer.lr))))
            # 色々初期化
            self.epoch = epoch
            self.epoch_start_time = time.time()
            self.reduce_on_epoch_end = False

        def on_batch_begin(self, batch, logs=None):
            pass

        def on_batch_end(self, batch, logs=None):
            logs = logs or {}
            loss = logs.get('loss')

            # nanチェック(一応)
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    print('Batch %d: Invalid loss, terminating training' % (batch))
                    self.model.stop_training = True

            # lossの指数移動平均の算出
            self.ema1 = loss * (1 - self.beta1) + self.ema1 * self.beta1
            self.ema2 = loss * (1 - self.beta2) + self.ema2 * self.beta2
            # Adam風補正
            self.iterations += 1
            hm1 = self.ema1 / (1 - self.beta1 ** self.iterations)
            hm2 = self.ema2 / (1 - self.beta2 ** self.iterations)
            self.delta_ema = hm2 - hm1
            if self.base_lr is not None:
                # lossの減少が止まってそうなら次のepochから学習率を減らす。
                self.iterations_per_reduce += 1
                if self.delta_ema <= 0 and self.margin_iterations <= self.iterations_per_reduce:
                    self.reduce_on_epoch_end = True

            # batchログ出力
            self.batch_log_writer.writerow([self.epoch + 1, batch + 1, '{:.4f}'.format(loss), '{:.5f}'.format(self.delta_ema)])

        def on_epoch_end(self, epoch, logs=None):
            # batchログ出力
            self.batch_log_file.flush()
            # epochログ出力
            lr = K.get_value(self.model.optimizer.lr)
            metrics = ['{:.4f}'.format(logs.get(k)) for k in self.params['metrics']]
            elapsed_time = time.time() - self.epoch_start_time
            self.epoch_log_writer.writerow([epoch + 1, '{:.1e}'.format(lr)] + metrics +
                                           [str(int(np.ceil(elapsed_time))), '{:.5f}'.format(self.delta_ema)])
            self.epoch_log_file.flush()
            # 学習率を減らす/限界まで下がっていたら学習終了
            if self.reduce_on_epoch_end:
                if self.max_reduces <= self.reduces:
                    # 限界まで下がっていたら学習終了
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                else:
                    # 学習率を減らす
                    self.reduces += 1
                    lr = self.base_lr * self.reduce_factor ** self.reduces
                    K.set_value(self.model.optimizer.lr, float(lr))
                    self.iterations_per_reduce = 0  # 安全装置のリセット
            if self.lr_list is not None and len(self.lr_list) - 1 <= epoch:
                # リストの最後まで来ていたら終了 (epochsをちゃんと設定すべきだが、安全装置として)
                self.stopped_epoch = epoch
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            self.batch_log_file.close()
            self.epoch_log_file.close()

    return _MyCallback


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
                config = tf.ConfigProto(gpu_options=tf.GPUOptions(**self.gpu_options), **self.config)
                K.set_session(tf.Session(config=config))

        def __exit__(self, *exc_info):
            if K.backend() == 'tensorflow':
                K.clear_session()

    return _Scope(config=config, gpu_options=gpu_options)


def learning_curve_plotter_factory():
    """Learning Curvesの描画を行う。

    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。
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
            self.filename = filename
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
                plt.savefig(str(self.filename).format(metric=self.metric))
                plt.close()

    return _LearningCurvePlotter


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

    def __init__(self, data_encoder=None, label_encoder=None, parallel=False):
        self.data_encoder = data_encoder
        self.label_encoder = label_encoder
        self.parallel = parallel

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, random_state=None, **kargs):
        """`fit_generator`などに渡すgenerator。kargsはそのままprepareに渡される。"""
        if self.parallel:
            with joblib.Parallel(n_jobs=batch_size, backend='threading') as parallel:
                return self._flow(X, y, weights, batch_size, shuffle, random_state, parallel, **kargs)
        else:
            return self._flow(X, y, weights, batch_size, shuffle, random_state, None, **kargs)

    def _flow(self, X, y, weights, batch_size, shuffle, random_state, parallel, **kargs):
        """`fit_generator`などに渡すgenerator。kargsはそのままprepareに渡される。"""
        random_state = sklearn.utils.check_random_state(random_state)
        length = len(X[0]) if isinstance(X, list) else len(X)

        if y is not None:
            assert length == (len(y[0]) if isinstance(y, list) else len(y))

        for ix in self._flow_indices(length, batch_size, shuffle, random_state):
            if isinstance(X, list):
                x_ = [t[ix] for t in X]  # multiple input
            else:
                x_ = X[ix]
            if y is not None:
                if isinstance(y, list):
                    y_ = [t[ix] for t in y]  # multiple output
                else:
                    y_ = y[ix]

            if y is None:
                assert weights is None
                yield self._prepare(x_, np.array([None] * len(x_)), np.array([None] * len(x_)), parallel, **kargs)[0]
            elif weights is None:
                yield self._prepare(x_, y_, np.array([None] * len(x_)), parallel, **kargs)[:2]
            else:
                yield self._prepare(x_, y_, weights[ix], parallel, **kargs)

    def _flow_indices(self, data_count, batch_size, shuffle, random_state=None):
        """データのindexを列挙し続けるgenerator。"""
        if shuffle:
            random_state = sklearn.utils.check_random_state(random_state)

        spe = self.steps_per_epoch(data_count, batch_size)
        ix = np.arange(data_count)
        while True:
            if shuffle:
                random_state.shuffle(ix)
            for bi in np.array_split(ix, spe):
                yield bi

    @staticmethod
    def steps_per_epoch(data_count, batch_size):
        """1epochが何ステップかを算出して返す"""
        return (data_count + batch_size - 1) // batch_size

    def _prepare(self, X, y, weights, parallel, **_):
        """何か前処理が必要な場合はこれをオーバーライドして使う。

        画像の読み込みとかDataAugmentationとか。
        yやweightsは使わない場合そのまま返せばOK。(使う場合はテスト時とかのNoneに注意。)
        """
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == weights.shape[0]
        assert self.parallel == (parallel is not None)
        if self.data_encoder:
            X = self.data_encoder(X)
        if self.label_encoder:
            y = self.label_encoder(y)
        return X, y, weights


def categorical_crossentropy(y_true, y_pred, alpha=0.75):
    """αによるclass=0とそれ以外の重み可変ありのcategorical_crossentropy。"""
    import keras.backend as K

    assert K.image_data_format() == 'channels_last'
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
        if 'keras_version' in f.attrs:
            original_keras_version = f.attrs['keras_version'].decode('utf8')
        else:
            original_keras_version = '1'
        if 'backend' in f.attrs:
            original_backend = f.attrs['backend'].decode('utf8')
        else:
            original_backend = None

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
