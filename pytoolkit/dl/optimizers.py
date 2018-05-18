"""Kerasのoptimizer関連。"""
import logging


def nsgd():
    """重み別に学習率の係数を設定できるSGD+Nesterov momentum Optimizer。"""
    import keras
    import keras.backend as K

    class NSGD(keras.optimizers.SGD):
        """重み別に学習率の係数を設定できるSGD+Nesterov momentum Optimizer。

        lr_multipliersは、Layerをキーとし、学習率の係数を値としたdict。

        # 例

        ```py
        lr_multipliers = {}
        lr_multipliers.update(zip(basenet.layers, [0.01] * len(basenet.layers)))
        ```

        """

        def __init__(self, lr=0.1, lr_multipliers=None, momentum=0.9, decay=0., nesterov=True, **kwargs):
            super().__init__(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, **kwargs)
            self.lr_multipliers = lr_multipliers

        @keras.legacy.interfaces.legacy_get_updates_support
        def get_updates(self, loss, params):
            applied_lr_multipliers = 0
            lr_multipliers = {}
            for layer, mp in (self.lr_multipliers or {}).items():
                for w in layer.trainable_weights:
                    lr_multipliers[w.name] = mp

            grads = self.get_gradients(loss, params)
            self.updates = []

            lr = self.lr
            if self.initial_decay > 0:
                lr *= (1. / (1. + self.decay * self.iterations))
                self.updates.append(K.update_add(self.iterations, 1))

            # momentum
            shapes = [K.get_variable_shape(p) for p in params]
            moments = [K.zeros(shape) for shape in shapes]
            self.weights = [self.iterations] + moments
            for p, g, m in zip(params, grads, moments):
                if p.name in lr_multipliers:
                    mlr = lr * lr_multipliers[p.name]
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

    return NSGD
