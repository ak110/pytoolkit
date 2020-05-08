"""Kerasのoptimizer関連。"""
# pylint: disable=no-name-in-module,attribute-defined-outside-init,invalid-unary-operand-type

import tensorflow as tf

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable()
class SGDEx(tf.keras.optimizers.SGD):
    """重み別に学習率の係数を設定できるSGD。

    lr_multipliersは、Layerまたは各weightのnameをキーとし、学習率の係数を値としたdict。

    例::

        lr_multipliers = {basenet: 0.1}

    """

    def __init__(
        self,
        learning_rate,
        lr_multipliers=None,
        momentum=0.9,
        decay=0.0,
        nesterov=True,
        lr=None,  # deprecated
        **kwargs,
    ):
        assert lr is None
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            decay=decay,
            nesterov=nesterov,
            **kwargs,
        )
        # {レイヤー: multiplier} or {重みの名前: multiplier}
        # model.save()時に前者はそのまま保存できないので、後者に統一する。
        self.lr_multipliers = {}
        for layer_or_weights_name, mp in (lr_multipliers or {}).items():
            if isinstance(layer_or_weights_name, str):
                self.lr_multipliers[layer_or_weights_name] = mp
            else:
                for w in layer_or_weights_name.trainable_weights:
                    self.lr_multipliers[w.name] = mp

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # pylint: disable=no-name-in-module,import-error
        from tensorflow.python.training import training_ops

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        if var.name in self.lr_multipliers:
            lr_t = coefficients["lr_t"] * self.lr_multipliers[var.name]
        else:
            lr_t = coefficients["lr_t"]

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            return training_ops.resource_apply_keras_momentum(
                var.handle,
                momentum_var.handle,
                lr_t,
                grad,
                coefficients["momentum"],
                use_locking=self._use_locking,
                use_nesterov=self.nesterov,
            )
        else:
            return training_ops.resource_apply_gradient_descent(
                var.handle, lr_t, grad, use_locking=self._use_locking
            )

    def get_config(self):
        config = {"lr_multipliers": self.lr_multipliers}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
