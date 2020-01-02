"""カスタムレイヤー。"""
import numpy as np
import tensorflow as tf

from .. import utils as tk_utils

K = tf.keras.backend


@tk_utils.register_keras_custom_object
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional Encodingレイヤー。

    x(i) = pos / pow(10000, 2 * i / depth)
    PE(pos, 2 * i) = sin(x(i))
    PE(pos, 2 * i + 1) = cos(x(i))

    ↑これを入力に足す。

    → 偶数・奇数で分けるのがやや面倒なので、depthの最初半分がsin, 後ろ半分がcosになるようにする
       && depthは偶数前提にしてしまう

    """

    def call(self, inputs, **kwargs):
        del kwargs
        _, max_length, depth = tf.unstack(K.shape(inputs))
        pos = K.cast(tf.range(max_length), K.floatx())
        i = K.cast(tf.range(depth // 2), K.floatx())
        d = K.cast(depth // 2, K.floatx())
        x_i = K.expand_dims(pos, -1) / K.expand_dims(
            K.pow(10000.0, 2.0 * i / d), 0
        )  # (max_length, depth // 2)
        pe0 = K.sin(x_i)
        pe1 = K.cos(x_i)
        pe = K.concatenate([pe0, pe1], axis=-1)  # (max_length, depth)
        pe = K.expand_dims(pe, axis=0)  # (1, max_length, depth)
        return inputs + pe


@tk_utils.register_keras_custom_object
class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head Attetion"""

    def __init__(
        self, units, heads=8, hidden_rate=1.0, drop_rate=0.1, causal=False, **kwargs
    ):
        super().__init__(**kwargs)
        assert units % heads == 0
        self.units = units
        self.heads = heads
        self.hidden_rate = hidden_rate
        self.drop_rate = drop_rate
        self.causal = causal
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.bq = None
        self.bk = None
        self.bv = None

    def compute_output_shape(self, input_shape):
        seq_shape, _ = input_shape
        return (seq_shape[0], seq_shape[1], self.units)

    def build(self, input_shape):
        seq_shape, ctx_shape = input_shape
        output_units = self.units // self.heads
        hidden_units = int(output_units * self.hidden_rate)
        self.Wq = self.add_weight(
            shape=(self.heads, int(seq_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            name="Wq",
        )
        self.Wk = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            name="Wk",
        )
        self.Wv = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), output_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            name="Wv",
        )
        self.bq = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            name="bq",
        )
        self.bk = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            name="bk",
        )
        self.bv = self.add_weight(
            shape=(self.heads, output_units),
            initializer=tf.keras.initializers.zeros(),
            name="bv",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        seq, ctx = inputs

        outputs = []
        for h in range(self.heads):
            # q.shape == (None, seq.shape[1], hidden_units)
            # k.shape == (None, ctx.shape[1], hidden_units)
            # v.shape == (None, ctx.shape[1], output_units)
            q = K.bias_add(K.dot(seq, self.Wq[h]), self.bq[h])
            k = K.bias_add(K.dot(ctx, self.Wk[h]), self.bk[h])
            v = K.bias_add(K.dot(ctx, self.Wv[h]), self.bv[h])
            k = k / np.sqrt(k.shape[-1])
            w = K.batch_dot(q, k, axes=(2, 2))  # (None, seq.shape[1], ctx.shape[1])
            if self.causal:
                w_shape = K.shape(w)
                mask_ones = tf.ones(shape=w_shape, dtype="int32")
                row_index = K.cumsum(mask_ones, axis=1)
                col_index = K.cumsum(mask_ones, axis=2)
                causal_mask = K.greater_equal(row_index, col_index)
                w = tf.where(causal_mask, w, K.tile([[[-np.inf]]], w_shape))
            w = K.softmax(w)
            w = K.dropout(w, level=self.drop_rate)  # Attention Dropout
            a = K.batch_dot(w, K.tanh(v), axes=(2, 1))
            # a.shape == (None, seq.shape[1], output_units)
            outputs.append(a)

        outputs = K.concatenate(outputs, axis=-1)
        return outputs

    def get_config(self):
        config = {
            "units": self.units,
            "heads": self.heads,
            "hidden_rate": self.hidden_rate,
            "drop_rate": self.drop_rate,
            "causal": self.causal,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tk_utils.register_keras_custom_object
class MultiHeadAttention2D(tf.keras.layers.Layer):
    """Multi-head Attetionの2D版のようなもの。(怪)"""

    def __init__(self, units, heads=8, hidden_rate=1.0, drop_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        assert units % heads == 0
        self.units = units
        self.heads = heads
        self.hidden_rate = hidden_rate
        self.drop_rate = drop_rate
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.bq = None
        self.bk = None
        self.bv = None

    def compute_output_shape(self, input_shape):
        seq_shape, _ = input_shape
        return (seq_shape[0], seq_shape[1], seq_shape[2], self.units)

    def build(self, input_shape):
        seq_shape, ctx_shape = input_shape
        output_units = self.units // self.heads
        hidden_units = int(output_units * self.hidden_rate)
        self.Wq = self.add_weight(
            shape=(self.heads, int(seq_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="Wq",
        )
        self.Wk = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), hidden_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="Wk",
        )
        self.Wv = self.add_weight(
            shape=(self.heads, int(ctx_shape[-1]), output_units),
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="Wv",
        )
        self.bq = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="bq",
        )
        self.bk = self.add_weight(
            shape=(self.heads, hidden_units),
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="bk",
        )
        self.bv = self.add_weight(
            shape=(self.heads, output_units),
            initializer=tf.keras.initializers.zeros(),
            regularizer=tf.keras.regularizers.l2(1e-4),
            name="bv",
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        del kwargs
        seq, ctx = inputs
        batch_size = K.shape(seq)[0]

        outputs = []
        for h in range(self.heads):
            # q.shape == (None, seq.shape[1], seq.shape[2], hidden_units)
            # k.shape == (None, ctx.shape[1], ctx.shape[2], hidden_units)
            # v.shape == (None, ctx.shape[1], ctx.shape[2], output_units)
            q = K.bias_add(K.dot(seq, self.Wq[h]), self.bq[h])
            k = K.bias_add(K.dot(seq, self.Wk[h]), self.bk[h])
            v = K.bias_add(K.dot(seq, self.Wv[h]), self.bv[h])
            q = K.reshape(q, (batch_size, -1, k.shape[-1]))
            k = K.reshape(k, (batch_size, -1, k.shape[-1]))
            v = K.reshape(v, (batch_size, -1, k.shape[-1]))
            k = k / np.sqrt(k.shape[-1])
            w = K.batch_dot(q, k, axes=(2, 2))  # (None, seq.shape[1], ctx.shape[1])
            w = K.softmax(w)
            w = K.dropout(w, level=self.drop_rate)  # Attention Dropout
            a = K.batch_dot(w, K.tanh(v), axes=(2, 1))
            # a.shape == (None, seq.shape[1], output_units)
            outputs.append(a)

        outputs = K.concatenate(outputs, axis=-1)
        output_shape = self.compute_output_shape([K.shape(seq), K.shape(ctx)])
        outputs = K.reshape(outputs, output_shape)
        return outputs

    def get_config(self):
        config = {
            "units": self.units,
            "heads": self.heads,
            "hidden_rate": self.hidden_rate,
            "drop_rate": self.drop_rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
