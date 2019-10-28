"""Kerasのoptimizer関連。"""
# pylint: disable=no-name-in-module,attribute-defined-outside-init,invalid-unary-operand-type

import tensorflow as tf

from . import utils as tk_utils

K = tf.keras.backend


@tk_utils.register_keras_custom_object
class SGDEx(tf.keras.optimizers.SGD):
    """重み別に学習率の係数を設定できるSGD。

    lr_multipliersは、Layerまたは各weightのnameをキーとし、学習率の係数を値としたdict。

    例::

        lr_multipliers = {basenet: 0.1}

    """

    def __init__(
        self,
        lr=None,  # deprecated
        learning_rate=0.1,
        lr_multipliers=None,
        momentum=0.9,
        decay=0.0,
        nesterov=True,
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


@tk_utils.register_keras_custom_object
class SAdam(tf.keras.optimizers.Optimizer):
    """色々混ぜてみたOptimizer。"""

    def __init__(
        self,
        lr=None,  # deprecated
        learning_rate=1e-3,
        beta_1=0.95,
        beta_2=0.999,
        l2=0,
        name="SAdam",
        **kwargs,
    ):
        assert lr is None
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("l2", l2)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")

    def _resource_apply_dense(self, grad, var):  # pylint: disable=arguments-differ
        var_dtype = var.dtype.base_dtype
        lr_t = self._get_hyper("learning_rate", var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        l2_t = self._get_hyper("l2", var_dtype)
        t = K.cast(self.iterations + 1, var_dtype)

        v_t = beta_2_t * v + (1 - beta_2_t) * K.square(grad)
        v_t = K.update(v, v_t)
        beta_2_pow_t = K.pow(beta_2_t, t)
        vhat_t = v_t / (1 - beta_2_pow_t)

        # AdamW
        grad += l2_t * var

        m_t = beta_1_t * m + grad  # velocity (like NovoGrad)
        m_t = K.update(m, m_t)

        # 重みの標準偏差によるスケーリング (怪)
        # Heの初期化では標準偏差が sqrt(2 / fan_in) で初期化されるので、
        # fan_in = 3のときstd = 0.81
        # fan_in = 256のときstd = 0.088
        # fan_in = 2048のときstd = 0.03125
        scale = K.clip(K.std(var), 0.01, 1.0) * 10

        if True:
            # RAdam

            rho_inf = 2 / (1 - beta_2_t) - 1
            rho_t = rho_inf - 2 * t * beta_2_pow_t / (1 - beta_2_pow_t)

            def update_adapted():
                r_t = K.sqrt(
                    ((rho_t - 4) * (rho_t - 2) * rho_inf)
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )
                return var - scale * lr_t * r_t * m_t / (K.sqrt(vhat_t) + 1e-5)

            def update_unadapted():
                return var - scale * lr_t * m_t

            var_t = K.switch(rho_t > 4, update_adapted, update_unadapted)
        else:
            var_t = var - scale * lr_t * m_t / (K.sqrt(vhat_t) + 1e-5)

        var_t = K.update(var, var_t)
        return tf.group(var_t, m_t, v_t)

    def _resource_apply_sparse(
        self, grad, var, indices
    ):  # pylint: disable=arguments-differ
        raise NotImplementedError("Sparse data is not supported yet")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "l2": self._serialize_hyperparameter("l2"),
            }
        )
        return config
