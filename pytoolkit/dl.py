"""DeepLearning(主にKeras)関連。

kerasをimportしてしまうとTensorFlowの初期化が始まって重いので、
importしただけではkerasがimportされないように作っている。

"""
import csv
import pathlib
import warnings

import numpy as np
import pandas as pd
import sklearn.utils


def destandarization_layer_factory():
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


def weighted_mean_layer_factory():
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


def get_custom_objects():
    """独自レイヤーのdictを返す。"""
    return {
        'Destandarization': destandarization_layer_factory(),
        'WeightedMean': weighted_mean_layer_factory(),
    }


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
                     max_reduces=6, reduce_factor=1 / np.sqrt(10),
                     beta1=0.998, beta2=0.999, margin_iterations=100):
            super().__init__()
            # 設定
            assert (lr_list is None) != (base_lr is None)  # どちらか片方のみ必須
            self.log_dir = log_dir
            self.batch_log_name = batch_log_name
            self.epoch_log_name = epoch_log_name
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
            self.keys = None
            self.iterations = 0
            self.iterations_per_reduce = 0
            self.ema1 = 0
            self.ema2 = 0
            self.reduces = 0
            self.stopped_epoch = 0
            self.epoch = 0
            self.reduce_on_epoch_end = False

        def on_train_begin(self, logs=None):
            # ログファイル作成
            d = pathlib.Path(self.log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.batch_log_file = d.joinpath(self.batch_log_name).open('w')
            self.epoch_log_file = d.joinpath(self.epoch_log_name).open('w')
            self.batch_log_writer = csv.writer(self.batch_log_file, delimiter='\t')
            self.epoch_log_writer = csv.writer(self.epoch_log_file, delimiter='\t')
            self.batch_log_writer.writerow(['epoch', 'batch', 'loss', 'delta_ema'])
            self.keys = None
            # 学習率の設定(base_lr)
            if self.base_lr is not None:
                K.set_value(self.model.optimizer.lr, float(self.base_lr))
                if self.verbose >= 1:
                    print('lr = {}'.format(float(K.get_value(self.model.optimizer.lr))))
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
                        print('lr = {}'.format(float(lr)))
                K.set_value(self.model.optimizer.lr, float(lr))
            elif self.reduce_on_epoch_end:
                if self.verbose >= 1:
                    print('lr = {}'.format(float(K.get_value(self.model.optimizer.lr))))
            # 色々初期化
            self.epoch = epoch
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
            delta_ema = hm2 - hm1
            if self.base_lr is not None:
                # lossの減少が止まってそうなら次のepochから学習率を減らす。
                self.iterations_per_reduce += 1
                if delta_ema <= 0 and self.margin_iterations <= self.iterations_per_reduce:
                    self.reduce_on_epoch_end = True

            # batchログ出力
            self.batch_log_writer.writerow([self.epoch + 1, batch + 1, loss, delta_ema])

        def on_epoch_end(self, epoch, logs=None):
            # batchログ出力
            self.batch_log_file.flush()
            # epochログ出力
            if not self.keys:
                self.keys = sorted(logs.keys())
                self.epoch_log_writer.writerow(['epoch', 'lr'] + self.keys)
            lr = K.get_value(self.model.optimizer.lr)
            metrics = [logs.get(k) for k in self.keys]
            self.epoch_log_writer.writerow([epoch + 1, lr] + metrics)
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
                K.set_session(
                    tf.Session(
                        config=tf.ConfigProto(
                            **self.config, gpu_options=tf.GPUOptions(**self.gpu_options))))

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
            except:
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
    df.plot(x='name', y='params', kind='barh', figsize=(16, 12))

    import matplotlib.pyplot as plt
    plt.gca().invert_yaxis()
    plt.savefig(str(to_file))
    plt.close()


def count_trainable_params(model):
    """modelのtrainable paramsを数える"""
    import keras.backend as K
    return sum([sum([K.count_params(p) for p in layer.trainable_weights]) for layer in model.layers])


class Generator(object):
    """`fit_generator`などに渡すgeneratorを作るためのベースクラス。"""

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, random_state=None, **kargs):
        """`fit_generator`などに渡すgenerator。kargsはそのままprepareに渡される。"""
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
                yield self._prepare(x_, **kargs)[0]
            elif weights is None:
                yield self._prepare(x_, y_, **kargs)[:2]
            else:
                yield self._prepare(x_, y_, weights[ix], **kargs)

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

    def _prepare(self, X, y=None, weights=None, **_):  # pylint: disable=no-self-use
        """何か前処理が必要な場合はこれをオーバーライドして使う。

        画像の読み込みとかDataAugmentationとか。
        yやweightsは使わない場合そのまま返せばOK。(使う場合はテスト時とかのNoneに注意。)
        """
        return X, y, weights
