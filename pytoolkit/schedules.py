"""学習率のスケジューリング。"""

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine Annealing without restart。

    Args:
        initial_learning_rate: 初期学習率
        decay_steps: 全体のステップ数 (len(train_set) // (batch_size * app.num_replicas_in_sync * tk.hvd.size()) * epochs)
        warmup_steps: 最初にlinear warmupするステップ数
        min_fraction: 初期学習率に対する最小の倍率
        name: 名前

    References:
        - SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>

    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        warmup_steps: int = 1000,
        min_fraction: float = 0.01,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.min_fraction = min_fraction
        self.name = name
        assert initial_learning_rate > 0
        assert 0 < warmup_steps < decay_steps
        assert min_fraction >= 0

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineAnnealing"):
            initial_learning_rate = tf.cast(
                self.initial_learning_rate, dtype=tf.float32,
            )
            decay_steps = tf.cast(self.decay_steps, tf.float32)
            warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            min_fraction = tf.cast(self.min_fraction, tf.float32)
            step = tf.cast(step, tf.float32)

            # linear warmup
            fraction1 = (step + 1) / warmup_steps

            # cosine annealing
            r = tf.math.minimum(step, decay_steps) / decay_steps
            fraction2 = 0.5 * (1.0 + tf.math.cos(np.pi * r))
            fraction2 = tf.math.maximum(fraction2, min_fraction)

            fraction = tf.where(step < warmup_steps, fraction1, fraction2)
            return tf.math.multiply(initial_learning_rate, fraction)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "min_fraction": self.min_fraction,
            "name": self.name,
        }
