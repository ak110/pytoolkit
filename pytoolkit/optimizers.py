"""Kerasのoptimizer関連。"""
# pylint: disable=no-name-in-module,attribute-defined-outside-init,invalid-unary-operand-type

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from . import K


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = []
    return {c.__name__: c for c in classes}


class SAdam(OptimizerV2):
    """色々混ぜてみたOptimizer。"""

    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.95,
        beta_2=0.999,
        l2=0,
        name="SAdam",
        **kwargs
    ):
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
