"""カスタムレイヤー。"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class DropActivation(tf.keras.layers.Layer):
    """Drop-Activation <https://arxiv.org/abs/1811.05850>"""

    def __init__(self, keep_rate=0.95, **kargs):
        super().__init__(**kargs)
        assert 0 <= keep_rate < 1
        self.keep_rate = keep_rate

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        def _train():
            shape = tf.shape(inputs)
            r = tf.random.uniform(shape=(shape[0],) + (1,) * (len(inputs.shape) - 1))
            return tf.where(r <= self.keep_rate, tf.nn.relu(inputs), inputs)

        def _test():
            return tf.nn.leaky_relu(inputs, alpha=1 - self.keep_rate)

        return tf.keras.backend.in_train_phase(_train, _test, training=training)

    def get_config(self):
        config = {"keep_rate": self.keep_rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
