"""Kerasのoptimizer関連。"""
# pylint: disable=cell-var-from-loop

from . import K, keras


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = [SRAdam]
    return {c.__name__: c for c in classes}


class SRAdam(keras.optimizers.Adam):
    """オレオレOptimizer。"""

    def __init__(self, lr=0.1, **kwargs):
        super().__init__(lr=lr, **kwargs)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # 重みの標準偏差によるスケーリング (怪)
            # Heの初期化では標準偏差が sqrt(2 / fan_in) で初期化されるので、
            # fan_in = 3のときstd = 0.81
            # fan_in = 256のときstd = 0.088
            # fan_in = 2048のときstd = 0.03125
            scale = K.clip(K.std(p), 0.01, 1.0)

            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(g)

            mhat_t = m_t / (1.0 - K.pow(self.beta_1, t))

            rho_inf = 2 / (1 - self.beta_2) - 1
            beta2t = K.pow(self.beta_2, t)
            rho_t = rho_inf - 2 * t * beta2t / (1 - beta2t)

            def update_adapted():
                vhat_t = K.sqrt(v_t / (1 - beta2t))
                r_t = K.sqrt(
                    ((rho_t - 4) * (rho_t - 2) * rho_inf)
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )
                return p - scale * self.lr * r_t * mhat_t / vhat_t

            def update_unadapted():
                return p - scale * self.lr * mhat_t

            p_t = K.switch(rho_t > 4, update_adapted, update_unadapted)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            if getattr(p, "constraint", None) is not None:
                p_t = p.constraint(p_t)
            self.updates.append(K.update(p, p_t))
        return self.updates
