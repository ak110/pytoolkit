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


@tf.keras.utils.register_keras_serializable()
class Clip(tf.keras.constraints.Constraint):
    """指定した値の範囲に制約する。"""

    def __init__(self, min_value, max_value, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        config = {"min_value": self.min_value, "max_value": self.max_value}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
