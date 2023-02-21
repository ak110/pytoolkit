"""tf.kerasのレイヤー関連。"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class CVPick(tf.keras.layers.Layer):
    """CVのモデルを一度に学習するための前処理レイヤー。

    学習時はfold_indexに一致しないところだけ出力。
    推論時はfold_indexに一致するところだけ出力。(out-of-fold prediction用)

    Examples:
        ::

            def _create_model(
                num_features: int, nfold: int
            ) -> tuple[tf.keras.models.Model, tf.keras.models.Model]:
                input_features = tf.keras.Input((num_features,), name="features")
                input_fold = tf.keras.Input((), dtype="int32", name="fold")

                fold_models = [
                    _create_fold_model(num_features, fold_index)
                    for fold_index in range(nfold)
                ]
                x = pytoolkit.tf.layers.CVMerge()(
                    [
                        m(pytoolkit.tf.layers.CVPick(fold_index)([
                            input_features, input_fold
                        ]))
                        for fold_index, m in enumerate(fold_models)
                    ]
                    + [input_fold]
                )
                train_model = tf.keras.models.Model([input_features, input_fold], x)

                x = tf.keras.layers.average([m(input_features) for m in fold_models])
                infer_model = tf.keras.models.Model(input_features, x)

                return train_model, infer_model

            def _create_fold_model(
                num_features: int, fold_index: int
            ) -> tf.keras.models.Model:
                inputs = x = tf.keras.Input((num_features,))
                x = tf.keras.layers.Dense(512, activation="gelu")(x)
                x += tf.keras.layers.Dense(512, activation="gelu")(x)
                x += tf.keras.layers.Dense(512, activation="gelu")(x)
                x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
                return tf.keras.models.Model(inputs, x,
                    name=f"fold_model_{fold_index + 1}"
                )


    """

    def __init__(self, fold_index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fold_index = fold_index

    def compute_output_shape(self, input_shape):
        """出力のshapeを返す。"""
        assert len(input_shape) == 2
        return input_shape[0]

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        """処理。"""
        x, fold = inputs

        def _train():
            return tf.boolean_mask(x, fold != self.fold_index, axis=0)

        def _test():
            return tf.boolean_mask(x, fold == self.fold_index, axis=0)

        return tf.keras.backend.in_train_phase(_train, _test, training)

    def get_config(self):
        """保存。"""
        config = {"fold_index": self.fold_index}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="pytoolkit")
class CVMerge(tf.keras.layers.Layer):
    """CVのモデルを一度に学習するための後処理レイヤー。

    学習時はfoldに一致しないところだけ集めて平均。
    推論時はfoldに一致するところだけ集めて出力。(out-of-fold prediction用)

    """

    def compute_output_shape(self, input_shape):
        """出力のshapeを返す。"""
        assert len(input_shape) >= 1
        return input_shape[0]

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        """処理。"""
        x_array, fold = inputs[:-1], inputs[-1]

        def _run(is_train):
            padded = [
                _scatter(is_train, x, fold, fold_index)
                for fold_index, x in enumerate(x_array)
            ]
            sum_ = tf.math.reduce_sum(tf.stack(padded, axis=0), axis=0)
            if is_train:
                return sum_ / (len(x_array) - 1)  # 重複数で割る
            else:
                return sum_  # 重複してないのでそのまま

        return tf.keras.backend.in_train_phase(
            lambda: _run(True), lambda: _run(False), training
        )


def _scatter(is_train, x, fold, fold_index):
    s = tf.concat([tf.shape(fold)[:1], tf.shape(x)[1:]], axis=0)
    z = tf.zeros(s, dtype=x.dtype)
    mask = (fold != fold_index) if is_train else (fold == fold_index)
    return tf.tensor_scatter_nd_update(z, tf.where(mask), x)
