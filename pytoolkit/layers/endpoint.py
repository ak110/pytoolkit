"""カスタムレイヤー。"""

import numpy as np
import tensorflow as tf

import pytoolkit as tk


@tf.keras.utils.register_keras_serializable()
class AutomatedFocalLoss(tf.keras.layers.Layer):
    """Automated Focal Loss <https://arxiv.org/abs/1904.09048>

    ラベルとsigmoid/softmaxの前の値を受け取り、損失の値を返す。

    Args:
        mode: "binary" or "categorical"
        class_weights: 各クラスの重み (不要ならNone)

    """

    def __init__(
        self, mode: str = "binary", class_weights: np.ndarray = None, **kwargs
    ):
        super().__init__(**kwargs)
        assert mode in ("binary", "categorical")
        self.mode = mode
        self.class_weights = class_weights
        self.phat_correct = None

    def build(self, input_shape):
        _, logits_shape = input_shape
        self.phat_correct = self.add_weight(
            shape=logits_shape[1:] if self.mode == "binary" else logits_shape[1:-1],
            initializer=tf.keras.initializers.zeros(),
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN,
            name="phat_correct",
        )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        _, logits_shape = input_shape
        if self.mode == "binary":
            return logits_shape
        else:
            return logits_shape[:-1]

    def call(self, inputs):
        labels, logits = inputs

        if self.mode == "binary":
            p = tf.math.sigmoid(logits)
            p_correct = labels * p + (1 - labels) * (1 - p)
        else:
            p = tf.nn.softmax(logits)
            p_correct = tf.math.reduce_sum(labels * p, axis=-1)

        p_new = tf.math.reduce_mean(p_correct, axis=0)
        phat_correct = self.phat_correct * 0.95 + p_new * 0.05
        tf.keras.backend.update(self.phat_correct, phat_correct)

        gamma = -tf.math.log(tf.math.maximum(phat_correct, 0.01))

        w = (1 - p_correct) ** gamma
        w = tf.stop_gradient(w)  # ？
        if self.mode == "binary":
            assert self.class_weights is None  # 未実装
            loss = w * (tf.math.log1p(tf.math.exp(logits)) - labels * logits)
        else:
            log_p = tk.backend.log_softmax(logits)
            base_loss = labels * log_p
            if self.class_weights is not None:
                base_loss = base_loss * self.class_weights
            loss = -w * tf.math.reduce_sum(base_loss, axis=-1)
        return loss

    def get_config(self):
        config = {"mode": self.mode, "class_weights": self.class_weights}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
