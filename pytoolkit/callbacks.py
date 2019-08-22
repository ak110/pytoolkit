"""DeepLearning(主にKeras)関連。"""
import csv
import pathlib
import time

import numpy as np

import pytoolkit as tk

from . import K, keras


class LearningRateStepDecay(keras.callbacks.Callback):
    """よくある150epoch目と225epoch目に学習率を1/10するコールバック。"""

    def __init__(self, reduce_epoch_rates=(0.5, 0.75), factor=0.1, epochs=None):
        super().__init__()
        self.reduce_epoch_rates = reduce_epoch_rates
        self.factor = factor
        self.epochs = epochs
        self.start_lr = None
        self.reduce_epochs = None

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.start_lr = float(K.get_value(self.model.optimizer.learning_rate))
        epochs = self.epochs or self.params["epochs"]
        self.reduce_epochs = [
            min(max(int(epochs * r), 1), epochs) for r in self.reduce_epoch_rates
        ]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch + 1 in self.reduce_epochs:
            lr1 = K.get_value(self.model.optimizer.learning_rate)
            lr2 = lr1 * self.factor
            K.set_value(self.model.optimizer.learning_rate, lr2)
            tk.log.get(__name__).info(
                f"Epoch {epoch + 1}: Learning rate {lr1:.1e} -> {lr2:.1e}"
            )

    def on_train_end(self, logs=None):
        # 終わったら戻しておく
        K.set_value(self.model.optimizer.learning_rate, self.start_lr)


class CosineAnnealing(keras.callbacks.Callback):
    """Cosine Annealing without restart。

    References:
        - SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>

    """

    def __init__(self, factor=0.01, epochs=None, warmup_epochs=5):
        assert factor < 1
        super().__init__()
        self.factor = factor
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.start_lr = None

    def on_train_begin(self, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.start_lr = float(K.get_value(self.model.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs=None):
        lr_max = self.start_lr
        lr_min = self.start_lr * self.factor
        if epoch + 1 <= self.warmup_epochs:
            r = (epoch + 1) / self.warmup_epochs
            lr = lr_min * (1 - r) + lr_max * r
        else:
            r = (epoch + 1) / (self.epochs or self.params["epochs"])
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * r))
        K.set_value(self.model.optimizer.learning_rate, float(lr))

    def on_train_end(self, logs=None):
        # 終わったら戻しておく
        K.set_value(self.model.optimizer.learning_rate, self.start_lr)


class TSVLogger(keras.callbacks.Callback):
    """ログを保存するコールバック。Horovod使用時はrank() == 0のみ有効。

    Args:
        filename: 保存先ファイル名。「{metric}」はmetricの値に置換される。str or pathlib.Path
        append: 追記するのか否か。

    """

    def __init__(self, filename, append=False, enabled=None):
        super().__init__()
        self.filename = pathlib.Path(filename)
        self.append = append
        self.enabled = enabled if enabled is not None else tk.hvd.is_master()
        self.log_file = None
        self.log_writer = None
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        if self.enabled:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = self.filename.open(
                "a" if self.append else "w", buffering=65536
            )
            self.log_writer = csv.writer(
                self.log_file, delimiter="\t", lineterminator="\n"
            )
            self.log_writer.writerow(
                ["epoch", "lr"] + self.params["metrics"] + ["time"]
            )
        else:
            self.log_file = None
            self.log_writer = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.learning_rate)
        elapsed_time = time.time() - self.epoch_start_time

        def _format_metric(logs, k):
            value = logs.get(k)
            if value is None:
                return "<none>"
            return f"{value:.4f}"

        metrics = [_format_metric(logs, k) for k in self.params["metrics"]]
        row = (
            [epoch + 1, format(logs["lr"], ".1e")]
            + metrics
            + [str(int(np.ceil(elapsed_time)))]
        )
        if self.log_file is not None:
            self.log_writer.writerow(row)
            self.log_file.flush()

    def on_train_end(self, logs=None):
        if self.log_file is not None:
            self.log_file.close()
        self.append = True  # 同じインスタンスの再利用時は自動的に追記にする


class EpochLogger(keras.callbacks.Callback):
    """DEBUGログを色々出力するcallback。Horovod使用時はrank() == 0のみ有効。"""

    def __init__(self, enabled=None):
        super().__init__()
        self.enabled = enabled if enabled is not None else tk.hvd.is_master()
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.learning_rate)
        now = time.time()
        elapsed_time = now - self.epoch_start_time
        time_per_epoch = (now - self.train_start_time) / (epoch + 1)
        eta = time_per_epoch * (self.params["epochs"] - epoch - 1)
        metrics = " ".join(
            [f"{k}={logs.get(k):.4f}" for k in self.params["metrics"] if k in logs]
        )
        if self.enabled:
            tk.log.get(__name__).debug(
                f"Epoch {epoch + 1:3d}: lr={lr:.1e} {metrics} time={int(np.ceil(elapsed_time))} ETA={int(np.ceil(eta))}"
            )


class FreezeBNCallback(keras.callbacks.Callback):
    """最後の指定epoch目でBNを全部freezeする。

    SENetの論文の最後の方にしれっと書いてあったので真似てみた。

    Args:
        freeze_epochs: BNをfreezeした状態で学習するepoch数。freeze_epochs=5なら最後の5epochをfreeze。

    使用例::

        callbacks.append(tk.dl.callbacks.freeze_bn(0.95))

    References:
        - Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>

    """

    def __init__(self, freeze_epochs: int):
        super().__init__()
        self.freeze_epochs = freeze_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch + 1 == self.params["epochs"] - self.freeze_epochs:
            freezed_count = self._freeze_layers(self.model)
            if freezed_count > 0:
                self._recompile()
            tk.log.get(__name__).info(
                f"Epoch {epoch + 1}: {freezed_count} BNs was frozen."
            )

    def _freeze_layers(self, container):
        freezed_count = 0
        for layer in container.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                if layer.trainable:
                    layer.trainable = False
                    freezed_count += 1
            elif hasattr(layer, "layers"):
                freezed_count += self._freeze_layers(layer)
        return freezed_count

    def _recompile(self):
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics,
            loss_weights=self.model.loss_weights,
            sample_weight_mode=self.model.sample_weight_mode,
        )


class UnfreezeCallback(keras.callbacks.Callback):
    """指定epoch目で全レイヤーをfreeze解除する。

    Args:
        epoch_rate: 発動するepoch数の割合を指定。

    使用例::

        callbacks.append(tk.dl.callbacks.unfreeze(0.1))

    """

    def __init__(self, epoch_rate: float):
        super().__init__()
        self.epoch_rate = epoch_rate
        self.target_epoch = 0
        assert 0 < epoch_rate <= 1

    def on_train_begin(self, logs=None):
        self.target_epoch = int(self.params["epochs"] * self.epoch_rate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.target_epoch == epoch + 1:
            unfreeze_count = self._unfreeze_layers(self.model)
            if unfreeze_count > 0:
                self._recompile()
            tk.log.get(__name__).info(
                f"Epoch {epoch + 1}: Unfreezed layers: {unfreeze_count}"
            )

    def _unfreeze_layers(self, container):
        count = 0
        for layer in container.layers:
            if not layer.trainable:
                layer.trainable = True
                count += 1
            elif hasattr(layer, "layers"):
                count += self._unfreeze_layers(layer)
        return count

    def _recompile(self):
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=self.model.loss,
            metrics=self.model.metrics,
            loss_weights=self.model.loss_weights,
            sample_weight_mode=self.model.sample_weight_mode,
            weighted_metrics=self.model.weighted_metrics,
        )


class Checkpoint(keras.callbacks.Callback):
    """学習中に定期的に保存する。

    速度重視でinclude_optimizerはFalse固定。

    Args:
        checkpoint_path: 保存先パス
        checkpoints: 保存する回数。epochs % (checkpoints + 1) == 0だとキリのいい感じになる。

    """

    def __init__(self, checkpoint_path, checkpoints=3):
        super().__init__()
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoints = checkpoints
        self.target_epochs = {}

    def on_train_begin(self, logs=None):
        s = self.checkpoints + 1
        self.target_epochs = {self.params["epochs"] * (i + 1) // s for i in range(s)}

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.target_epochs:
            if tk.hvd.is_master():
                tk.log.get(__name__).info(
                    f"Epoch {epoch}: Saving model to {self.checkpoint_path}"
                )
                # 保存の真っ最中に死んだときに大丈夫なように少し気を使った処理をする。
                # 一度.tmpに保存してから元のを.bkにリネームして.tmpを外して.bkを削除。
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.checkpoint_path.parent / (
                    self.checkpoint_path.name + ".tmp"
                )
                backup_path = self.checkpoint_path.parent / (
                    self.checkpoint_path.name + ".bk"
                )
                if temp_path.exists():
                    temp_path.unlink()
                if backup_path.exists():
                    backup_path.unlink()
                self.model.save(str(temp_path), include_optimizer=False)
                if self.checkpoint_path.exists():
                    self.checkpoint_path.rename(backup_path)
                temp_path.rename(self.checkpoint_path)
                if backup_path.exists():
                    backup_path.unlink()
            tk.hvd.barrier()


class ErrorOnNaN(keras.callbacks.Callback):
    """NaNやinfで異常終了させる。"""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                raise RuntimeError(f"Batch {batch}: Invalid loss")
