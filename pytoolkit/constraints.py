"""制約。"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class GreaterThanOrEqualTo(tf.keras.constraints.Constraint):
    """指定した値以上に制約する。"""

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def __call__(self, w):
        return tf.math.maximum(w, self.value)

    def get_config(self):
        config = {"value": self.value}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
