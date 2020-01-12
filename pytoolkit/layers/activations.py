"""カスタムレイヤー。"""
import tensorflow as tf

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable()
class TLU(tf.keras.layers.Layer):
    """Thresholded Linear Unit <https://arxiv.org/abs/1911.09737>"""

    def __init__(
        self,
        tau_initializer="zeros",
        tau_regularizer=None,
        tau_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)
        self.tau = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        self.tau = self.add_weight(
            shape=affine_shape,
            name="tau",
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        return tf.maximum(inputs, self.tau)

    def get_config(self):
        config = {
            "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer),
            "tau_regularizer": tf.keras.regularizers.serialize(self.tau_regularizer),
            "tau_constraint": tf.keras.constraints.serialize(self.tau_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

        return K.in_train_phase(_train, _test, training=training)

    def get_config(self):
        config = {"keep_rate": self.keep_rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
