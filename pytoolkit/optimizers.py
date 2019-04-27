"""Kerasのoptimizer関連。"""
import logging

from . import K, keras


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = [
        NSGD,
        SAdam,
    ]
    return {c.__name__: c for c in classes}


class NSGD(keras.optimizers.SGD):
    """重み別に学習率の係数を設定できるSGD+Nesterov momentum Optimizer。

    lr_multipliersは、Layerをキーとし、学習率の係数を値としたdict。

    例::

        lr_multipliers = {}
        lr_multipliers.update(zip(basenet.layers, [0.01] * len(basenet.layers)))

    """

    def __init__(self, lr=0.1, lr_multipliers=None, momentum=0.9, decay=0., nesterov=True, **kwargs):
        super().__init__(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs)
        # {レイヤー: multiplier} or {重みの名前: multiplier}
        # model.save()時に前者はそのまま保存できないので、後者に統一する。
        self.lr_multipliers = {}
        for layer_or_weights_name, mp in (lr_multipliers or {}).items():
            if isinstance(layer_or_weights_name, str):
                self.lr_multipliers[layer_or_weights_name] = mp
            else:
                for w in layer_or_weights_name.trainable_weights:
                    self.lr_multipliers[w.name] = mp
        self.updates = None
        self.weights = None

    def get_updates(self, loss, params):
        applied_lr_multipliers = 0

        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * self.iterations))

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            if p.name in self.lr_multipliers:
                mlr = lr * self.lr_multipliers[p.name]
                applied_lr_multipliers += 1
            else:
                mlr = lr
            v = self.momentum * m - mlr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - mlr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        logger = logging.getLogger(__name__)
        logger.info(f'lr_multipliers: applied = {applied_lr_multipliers}')
        return self.updates

    def get_config(self):
        config = {'lr_multipliers': self.lr_multipliers}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SAdam(keras.optimizers.Adam):
    """オレオレOptimizer。

    lr_multipliersは、Layerをキーとし、学習率の係数を値としたdict。

    例::

        lr_multipliers = {}
        lr_multipliers.update(zip(basenet.layers, [0.01] * len(basenet.layers)))

    """

    def __init__(self, lr=0.1, lr_multipliers=None, **kwargs):
        super().__init__(lr=lr, **kwargs)
        # {レイヤー: multiplier} or {重みの名前: multiplier}
        # model.save()時に前者はそのまま保存できないので、後者に統一する。
        self.lr_multipliers = {}
        for layer_or_weights_name, mp in (lr_multipliers or {}).items():
            if isinstance(layer_or_weights_name, str):
                self.lr_multipliers[layer_or_weights_name] = mp
            else:
                for w in layer_or_weights_name.trainable_weights:
                    self.lr_multipliers[w.name] = mp
        self.updates = None
        self.weights = None

    def get_updates(self, loss, params):
        applied_lr_multipliers = 0

        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        # lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / 1. - K.pow(self.beta_1, t))
        # あえてbeta_1の方は削除してみる (怪)
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            # 学習率の個別調整
            if p.name in self.lr_multipliers:
                mlr = lr_t * self.lr_multipliers[p.name]
                applied_lr_multipliers += 1
            else:
                mlr = lr_t

            # 重みの標準偏差によるスケーリング (怪)
            # Heの初期化では標準偏差が sqrt(2 / fan_in) で初期化されるので、
            # fan_in = 3のときstd = 0.81
            # fan_in = 256のときstd = 0.088
            # fan_in = 2048のときstd = 0.03125
            scale = K.maximum(K.std(p), 1e-2)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - mlr * scale * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - scale * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr_multipliers': self.lr_multipliers}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
