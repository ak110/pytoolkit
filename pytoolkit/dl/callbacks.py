"""DeepLearning(主にKeras)関連。"""
import csv
import pathlib
import time
import warnings

import numpy as np

from . import hvd
from .. import draw, io, log


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
            # 重複は禁止
            assert len(self.reduce_epochs) == len(np.unique(self.reduce_epochs)), f'reduce_epochsエラー: {self.reduce_epochs}'
            # Horovod使用時はWarmupとぶつかりかねないので5epoch以下でのreduceは禁止
            assert not (hvd.initialized() and self.reduce_epochs[0] <= 5), f'reduce_epochsエラー: {self.reduce_epochs}'

        def on_epoch_begin(self, epoch, logs=None):
            if epoch + 1 in self.reduce_epochs:
                lr1 = K.get_value(self.model.optimizer.lr)
                lr2 = lr1 * self.factor
                K.set_value(self.model.optimizer.lr, lr2)
                logger = log.get(self.logger_name or __name__)
                logger.info(f'Epoch {epoch + 1}: Learning rate {lr1:.1e} -> {lr2:.1e}')

    return _LearningRate(reduce_epoch_rates=reduce_epoch_rates, factor=factor, logger_name=logger_name)


def cosine_annealing(factor=0.01):
    """Cosine Annealing without restart。

    ■SGDR: Stochastic Gradient Descent with Warm Restarts
    https://arxiv.org/abs/1608.03983
    """
    import keras
    import keras.backend as K
    assert factor < 1

    class _CosineAnnealing(keras.callbacks.Callback):

        def __init__(self, factor):
            self.factor = factor
            self.period = None
            self.start_lr = None
            super().__init__()

        def on_train_begin(self, logs=None):
            self.start_lr = float(K.get_value(self.model.optimizer.lr))

        def on_epoch_begin(self, epoch, logs=None):
            lr_max = self.start_lr
            lr_min = self.start_lr * self.factor
            r = (epoch + 1) / self.params['epochs']
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * r))
            K.set_value(self.model.optimizer.lr, float(lr))

    return _CosineAnnealing(factor=factor)


def learning_curve_plot(filename, metric='loss'):
    """Learning Curvesの描画を行う。

    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - metric: 対象とするmetric名。lossとかaccとか。

    """
    import keras

    class _LearningCurvePlotter(keras.callbacks.Callback):

        def __init__(self, filename, metric='loss'):
            self.filename = pathlib.Path(filename).resolve()
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

                with draw.get_lock():
                    ax = df.plot()
                    draw.save(ax, self.filename.parent / self.filename.name.format(metric=self.metric))
                    draw.close(ax)

    return _LearningCurvePlotter(filename=filename, metric=metric)


def tsv_logger(filename, append=False):
    """ログを保存するコールバック。Horovod使用時はrank() == 0のみ有効。

    # 引数
    - filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
    - append: 追記するのか否か。

    """
    import keras
    import keras.backend as K

    enabled = hvd.is_master()

    class _TSVLogger(keras.callbacks.Callback):

        def __init__(self, filename, append, enabled):
            self.filename = pathlib.Path(filename)
            self.append = append
            self.enabled = enabled
            self.log_file = None
            self.log_writer = None
            self.epoch_start_time = None
            super().__init__()

        def on_train_begin(self, logs=None):
            if self.enabled:
                self.filename.parent.mkdir(parents=True, exist_ok=True)
                self.log_file = self.filename.open('a' if self.append else 'w', buffering=65536)
            else:
                self.log_file = io.open_devnull('w', buffering=65536)
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
            self.log_writer.writerow([epoch + 1, format(logs['lr'], '.1e')] + metrics +
                                     [str(int(np.ceil(elapsed_time)))])
            self.log_file.flush()

        def on_train_end(self, logs=None):
            self.log_file.close()

    return _TSVLogger(filename=filename, append=append, enabled=enabled)


def epoch_logger(logger_name=None):
    """`logging.getLogger(logger_name)`にDEBUGログを色々出力するcallback。Horovod使用時はrank() == 0のみ有効。"""
    import keras
    import keras.backend as K

    enabled = hvd.is_master()

    class _EpochLogger(keras.callbacks.Callback):

        def __init__(self, logger_name, enabled):
            self.logger_name = logger_name
            self.enabled = enabled
            self.logger = log.get(self.logger_name or __name__)
            self.epoch_start_time = None
            super().__init__()

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            lr = K.get_value(self.model.optimizer.lr)
            elapsed_time = time.time() - self.epoch_start_time
            metrics = ' '.join([f'{k}={logs.get(k):.4f}' for k in self.params['metrics'] if k in logs])
            if enabled:
                self.logger.debug('Epoch %3d: lr=%.1e %s time=%d',
                                  epoch + 1, lr, metrics, int(np.ceil(elapsed_time)))

    return _EpochLogger(logger_name=logger_name, enabled=enabled)


def freeze_bn(freeze_epoch_rate: float, logger_name=None):
    """指定epoch目でBNを全部freezeする。

    SENetの論文の最後の方にしれっと書いてあったので真似てみた。

    ■Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507

    # 引数
    - freeze_epoch_rate: 発動するepoch数の割合を指定。
    - logger_name: ログ出力先

    # 使用例

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
                logger.info(f'Epoch {epoch + 1}: Freeze BNs (layers = {len(self.freezed_layers)})')

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
            logger.info(f'Epoch {self.params["epochs"]}: Unfreeze BNs (layers = {unfreezed})')

        def _recompile(self):
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics,
                loss_weights=self.model.loss_weights,
                sample_weight_mode=self.model.sample_weight_mode,
                weighted_metrics=self.model.weighted_metrics)

    return _FreezeBNCallback(freeze_epoch_rate=freeze_epoch_rate, logger_name=logger_name)
