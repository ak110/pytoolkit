"""カスタムレイヤー。"""
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class SyncBatchNormalization(tf.keras.layers.Layer):
    """Sync BN。"""

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(
            moving_mean_initializer
        )
        self.moving_variance_initializer = tf.keras.initializers.get(
            moving_variance_initializer
        )
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init

        target_axes = self._get_target_axes(ndims=len(input_shape))
        param_shape = [s for i, s in enumerate(input_shape) if i in target_axes]

        if self.scale:
            self.gamma = self.add_weight(
                shape=param_shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=param_shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=param_shape,
            dtype=tf.float32,
            initializer=self.moving_mean_initializer,
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            experimental_autocast=False,
        )

        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=param_shape,
            dtype=tf.float32,
            initializer=self.moving_variance_initializer,
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_WRITE,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            experimental_autocast=False,
        )

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        del kwargs
        return tf.keras.backend.in_train_phase(
            lambda: self._bn_train(inputs),
            lambda: self._bn_test(inputs),
            training if self.trainable else False,
        )

    def _bn_train(self, inputs):
        """学習時のBN。"""
        # self.axisを除く軸で平均・分散を算出する
        target_axes = self._get_target_axes(ndims=inputs.shape.rank)
        stat_axes = [a for a in range(inputs.shape.rank) if a not in target_axes]

        # 平均・分散の算出
        x = tf.cast(inputs, tf.float32)
        if tf.version.VERSION.startswith("2.2"):  # workaround
            x = tf.debugging.assert_all_finite(x, "x")
        mean = tf.math.reduce_mean(x, axis=stat_axes)
        squared_mean = tf.math.reduce_mean(tf.math.square(x), axis=stat_axes)
        # if tf.version.VERSION.startswith("2.2"):  # workaround
        mean = tf.debugging.assert_all_finite(mean, "mean")
        squared_mean = tf.debugging.assert_all_finite(squared_mean, "squared_mean")

        # Sync BN
        if tk.hvd.initialized():
            import horovod.tensorflow as _hvd

            mean = _hvd.allreduce(mean, op=_hvd.Average)
            squared_mean = _hvd.allreduce(squared_mean, op=_hvd.Average)
        else:
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                mean = replica_context.all_reduce(tf.distribute.ReduceOp.MEAN, mean)
                squared_mean = replica_context.all_reduce(
                    tf.distribute.ReduceOp.MEAN, squared_mean
                )
            else:
                strategy = tf.distribute.get_strategy()
                mean = strategy.reduce(tf.distribute.ReduceOp.MEAN, mean, axis=None)
                squared_mean = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, squared_mean, axis=None
                )

        var = squared_mean - tf.math.square(mean)
        if tf.version.VERSION.startswith("2.2"):  # workaround
            mean = tf.debugging.assert_all_finite(mean, "reduced mean")
            var = tf.debugging.assert_all_finite(var, "reduced var")

        # exponential moving average:
        # m_new = m_old * 0.99 + x * 0.01
        # m_new - m_old = (x - m_old) * 0.01
        decay = 1 - self.momentum
        self.add_update(
            [
                self.moving_mean.assign_add(
                    (mean - self.moving_mean) * decay,
                    read_value=False,
                ),
                self.moving_variance.assign_add(
                    (var - self.moving_variance) * decay,
                    read_value=False,
                ),
            ]
        )

        # y = (x - mean) / (sqrt(var) + epsilon) * gamma + beta
        #   = x * gamma / (sqrt(var) + epsilon) + (beta - mean * gamma / (sqrt(var) + epsilon))
        #   = x * a + (beta - mean * a)
        a = self.gamma * tf.math.rsqrt(var + self.epsilon)
        b = self.beta - mean * a
        return K.cast(x * a + b, K.dtype(inputs))

    def _bn_test(self, inputs):
        """予測時のBN。"""
        x = tf.cast(inputs, tf.float32)
        a = self.gamma * tf.math.rsqrt(self.moving_variance + self.epsilon)
        b = self.beta - self.moving_mean * a
        return tf.cast(x * a + b, inputs.dtype)

    def _get_target_axes(self, ndims):
        axes = [self.axis] if isinstance(self.axis, int) else list(self.axis)
        for idx, x in enumerate(axes):
            if x < 0:
                axes[idx] = ndims + x
        return axes

    def get_config(self):
        config = {
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "moving_mean_initializer": tf.keras.initializers.serialize(
                self.moving_mean_initializer
            ),
            "moving_variance_initializer": tf.keras.initializers.serialize(
                self.moving_variance_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class RMSNormalization(tf.keras.layers.Layer):
    """Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>"""

    def __init__(
        self,
        axis=-1,
        epsilon=1e-7,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        if self.scale:
            self.gamma = self.add_weight(
                shape=affine_shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=affine_shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs

        ms = tf.math.reduce_mean(inputs ** 2, axis=self.axis, keepdims=True)
        outputs = inputs * tf.math.rsqrt(ms + self.epsilon)

        broadcast_shape = (1,) * (K.ndim(inputs) - 1) + (-1,)
        if self.scale:
            outputs = outputs * tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            outputs = outputs + tf.reshape(self.beta, broadcast_shape)

        return outputs

    def get_config(self):
        config = {
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class RandomRMSNormalization(tf.keras.layers.Layer):
    """ランダム要素のあるrmsを使ったnormalization。<https://twitter.com/ak11/status/1202838201716490240>"""

    def __init__(
        self,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.beta = None
        self.gamma = None

    def build(self, input_shape):
        affine_shape = (input_shape[-1],)
        if self.scale:
            self.gamma = self.add_weight(
                shape=affine_shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        if self.center:
            self.beta = self.add_weight(
                shape=affine_shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        del kwargs
        axes = list(range(1, inputs.shape.rank - 1))

        nu2 = tf.math.reduce_mean(tf.math.square(inputs), axis=axes, keepdims=True)

        #  学習時はノイズを入れてみる
        nu2 += K.in_train_phase(tf.random.normal((), stddev=0.02), 0.0, training)

        x = inputs * tf.math.rsqrt(nu2 + 1.0)  # >= 1にするため+1
        if self.scale:
            x *= self.gamma
        if self.center:
            x += self.beta
        return x

    def get_config(self):
        config = {
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
