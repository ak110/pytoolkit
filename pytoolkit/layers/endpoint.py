"""カスタムレイヤー。"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class AutomatedFocalLoss(tf.keras.layers.Layer):
    """Automated Focal Loss <https://arxiv.org/abs/1904.09048>

    ラベルとsigmoid/softmaxの前の値を受け取り、損失の値を返す。

    Args:
        mode: "binary" or "categorical"

    """

    def __init__(self, mode: str = "binary", **kwargs):
        super().__init__(**kwargs)
        assert mode in ("binary", "categorical")
        self.mode = mode
        self.phat_correct = None

    def build(self, input_shape):
        _, logits_shape = input_shape
        self.phat_correct = self.add_weight(
            shape=logits_shape[1:] if self.mode == "binary" else logits_shape[1:-1],
            initializer=tf.keras.initializers.zeros(),
            trainable=False,
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

        new = tf.math.reduce_mean(p_correct, axis=0)
        phat_correct = self.phat_correct * 0.95 + new * 0.05
        tf.keras.backend.update(self.phat_correct, phat_correct)

        gamma = -tf.math.log(phat_correct + 1e-7)

        w = (1 - p_correct) ** gamma
        loss = -w * tf.math.log(p_correct + 1e-7)
        return loss

    def get_config(self):
        config = {"mode": self.mode}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
