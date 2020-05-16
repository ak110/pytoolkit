"""カスタムレイヤー。"""
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


@tf.keras.utils.register_keras_serializable()
class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
    """Sync BN。"""

    def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
        del kwargs
        return K.in_train_phase(
            lambda: self._bn_train(inputs), lambda: self._bn_test(inputs), training
        )

    def _bn_train(self, inputs):
        """学習時のBN。"""
        # self.axisを除く軸で平均・分散を算出する
        target_axis = self.axis
        if isinstance(target_axis, int):
            target_axis = [target_axis]
        stat_axes = [a for a in range(K.ndim(inputs)) if a not in target_axis]

        # 平均・分散の算出
        x = inputs if K.dtype(inputs) == "float32" else K.cast(inputs, "float32")
        mean = K.mean(x, axis=stat_axes)
        squared_mean = K.mean(K.square(x), axis=stat_axes)
        # Sync BN
        if tk.hvd.initialized():
            import horovod.tensorflow as _hvd

            mean = _hvd.allreduce(mean, average=True)
            squared_mean = _hvd.allreduce(squared_mean, average=True)
        var = squared_mean - K.square(mean)

        # exponential moving average:
        # m_new = m_old * 0.99 + x * 0.01
        # m_new - m_old = (x - m_old) * 0.01
        decay = 1 - self.momentum
        update1 = tf.compat.v1.assign_add(
            self.moving_mean, (mean - self.moving_mean) * decay
        )
        update2 = tf.compat.v1.assign_add(
            self.moving_variance, (var - self.moving_variance) * decay
        )
        self.add_update([update1, update2], inputs)

        # y = (x - mean) / (sqrt(var) + epsilon) * gamma + beta
        #   = x * gamma / (sqrt(var) + epsilon) + (beta - mean * gamma / (sqrt(var) + epsilon))
        #   = x * a + (beta - mean * a)
        a = self.gamma * tf.math.rsqrt(var + 1e-7)
        b = self.beta - mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))

    def _bn_test(self, inputs):
        """予測時のBN。"""
        a = self.gamma / tf.math.rsqrt(self.moving_variance + 1e-7)
        b = self.beta - self.moving_mean * a
        return inputs * K.cast(a, K.dtype(inputs)) + K.cast(b, K.dtype(inputs))


@tf.keras.utils.register_keras_serializable()
class GroupNormalization(tf.keras.layers.Layer):
    """Group Normalization。

    Args:
        groups: グループ数

    References:
        - Group Normalization <https://arxiv.org/abs/1803.08494>

    """

    def __init__(
        self,
        groups=32,
        epsilon=1e-5,
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
        self.groups = groups
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        dim = int(input_shape[-1])
        groups = min(dim, self.groups)
        assert dim % groups == 0
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        del kwargs
        x = inputs
        ndim = K.ndim(x)
        shape = K.shape(x)
        if ndim == 4:  # 2D
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            g = K.minimum(self.groups, C)
            x = tf.reshape(x, [N, H, W, g, C // g])
            mean, var = tf.nn.moments(x=x, axes=[1, 2, 4], keepdims=True)
            x = (x - mean) * tf.math.rsqrt(var + self.epsilon)
            x = tf.reshape(x, [N, H, W, C])
        elif ndim == 5:  # 3D
            N, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]
            g = K.minimum(self.groups, C)
            x = tf.reshape(x, [N, T, H, W, g, C // g])
            mean, var = tf.nn.moments(x=x, axes=[1, 2, 3, 5], keepdims=True)
            x = (x - mean) * tf.math.rsqrt(var + self.epsilon)
            x = tf.reshape(x, [N, T, H, W, C])
        else:
            assert ndim in (4, 5)
        if self.scale:
            x = x * self.gamma
        if self.center:
            x = x + self.beta
        # tf.keras用
        x.set_shape(inputs.shape.as_list())
        return x

    def get_config(self):
        config = {
            "groups": self.groups,
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


@tf.keras.utils.register_keras_serializable()
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization"""

    def __init__(
        self,
        epsilon=1e-5,
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
        input_shape = inputs.shape.as_list()

        reduction_axes = list(range(1, len(input_shape) - 1))
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        std = K.std(inputs, reduction_axes, keepdims=True)
        outputs = (inputs - mean) / (std + self.epsilon)

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[-1] = input_shape[-1]
        if self.scale:
            outputs = outputs * tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            outputs = outputs + tf.reshape(self.beta, broadcast_shape)

        return outputs

    def get_config(self):
        config = {
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


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
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
