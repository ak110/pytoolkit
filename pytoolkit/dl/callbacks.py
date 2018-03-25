"""DeepLearning(主にKeras)関連。"""
import csv
import pathlib
import time
import warnings

import numpy as np

from .. import log


def learning_rate(reduce_epoch_rates=(0.5, 0.75), factor=0.1, logger_name=None):
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


def learning_curve_plot(filename, metric='loss'):
    """Learning Curvesの描画を行う。

    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - metric: 対象とするmetric名。lossとかaccとか。

    # 「Invalid DISPLAY variable」対策

    環境変数MPLBACKEND=Aggを設定しておく。
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
                import pandas as pd
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


def tsv_logger(filename, append=False):
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


def logger(name='__main__'):
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


def freeze_bn(freeze_epoch_rate: float, logger_name=None):
    """指定epoch目でBNを全部freezeする。

    SENetの論文の最後の方にしれっと書いてあったので真似てみた。

    ■Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507

    ## 引数
    - freeze_epoch_rate: 発動するepoch数の割合を指定。
    - logger_name: ログ出力先

    ## 使用例

    ```
    callbacks.append(tk.dl.callbacks.freeze_bn(0.95))
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

    import pandas as pd
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

        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]  # pylint: disable=not-an-iterable

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
