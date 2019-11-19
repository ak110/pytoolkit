"""DeepLearning(主にKeras)関連。"""
import pathlib
import time

import numpy as np
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


class LearningRateStepDecay(tf.keras.callbacks.Callback):
    """よくある150epoch目と225epoch目に学習率を1/10するコールバック。"""

    def __init__(self, reduce_epoch_rates=(0.5, 0.75), factor=0.1, epochs=None):
        super().__init__()
        self.reduce_epoch_rates = reduce_epoch_rates
        self.factor = factor
        self.epochs = epochs
        self.start_lr = None
        self.reduce_epochs = None

    def on_train_begin(self, logs=None):
        del logs
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.start_lr = float(K.get_value(self.model.optimizer.learning_rate))
        epochs = self.epochs or self.params["epochs"]
        self.reduce_epochs = [
            min(max(int(epochs * r), 1), epochs) for r in self.reduce_epoch_rates
        ]

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        if epoch + 1 in self.reduce_epochs:
            lr1 = K.get_value(self.model.optimizer.learning_rate)
            lr2 = lr1 * self.factor
            K.set_value(self.model.optimizer.learning_rate, lr2)
            tk.log.get(__name__).info(
                f"Epoch {epoch + 1}: Learning rate {lr1:.1e} -> {lr2:.1e}"
            )

    def on_train_end(self, logs=None):
        del logs
        # 終わったら戻しておく
        K.set_value(self.model.optimizer.learning_rate, self.start_lr)


class CosineAnnealing(tf.keras.callbacks.Callback):
    """Cosine Annealing without restart。

    References:
        - SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>

    """

    def __init__(
        self, factor=0.01, epochs=None, warmup_epochs=5, warmup_reset_state=True
    ):
        assert factor < 1
        super().__init__()
        self.factor = factor
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_reset_state = warmup_reset_state
        self.start_lr = None

    def on_train_begin(self, logs=None):
        del logs
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        self.start_lr = float(K.get_value(self.model.optimizer.learning_rate))

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        lr_max = self.start_lr
        lr_min = self.start_lr * self.factor
        if epoch + 1 < self.warmup_epochs:
            learning_rate = lr_max * (epoch + 1) / self.warmup_epochs
        else:
            r = (epoch + 1) / (self.epochs or self.params["epochs"])
            learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * r))
        K.set_value(self.model.optimizer.learning_rate, float(learning_rate))
        # warmup時、optimizerの状態をゼロ埋めする (SGDなど前提で種類無視して0にするので注意)
        # https://twitter.com/ak11/status/1194763993082023936
        if self.warmup_reset_state and 2 <= epoch + 1 <= self.warmup_epochs:
            state = self.model.optimizer.get_weights()
            self.model.optimizer.set_weights([np.zeros_like(w) for w in state])

    def on_train_end(self, logs=None):
        del logs
        # 終わったら戻しておく
        K.set_value(self.model.optimizer.learning_rate, self.start_lr)


class EpochLogger(tf.keras.callbacks.Callback):
    """DEBUGログを色々出力するcallback。Horovod使用時はrank() == 0のみ有効。"""

    def __init__(self, enabled=None):
        super().__init__()
        self.enabled = enabled if enabled is not None else tk.hvd.is_master()
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        del logs
        self.train_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        del epoch, logs
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        assert self.train_start_time is not None
        assert self.epoch_start_time is not None
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
                f"Epoch {epoch + 1:3d}: learning_rate={lr:.1e} {metrics} time={int(np.ceil(elapsed_time))} ETA={int(np.ceil(eta))}"
            )


class Checkpoint(tf.keras.callbacks.Callback):
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
        del logs
        s = self.checkpoints + 1
        self.target_epochs = {self.params["epochs"] * (i + 1) // s for i in range(s)}

    def on_epoch_begin(self, epoch, logs=None):
        del logs
        if epoch in self.target_epochs:
            if tk.hvd.is_master():
                tk.log.get(__name__).info(
                    f"Epoch {epoch}: Saving model to {self.checkpoint_path}"
                )
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.checkpoint_path))
            tk.hvd.barrier()


class ErrorOnNaN(tf.keras.callbacks.Callback):
    """NaNやinfで異常終了させる。"""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                raise RuntimeError(f"Batch {batch}: Invalid loss")
