"""テストコード。"""

import tensorflow as tf

import pytoolkit.tf


def test_CVPick():
    layer = pytoolkit.tf.layers.CVPick(fold_index=1)

    x = tf.reshape(tf.range(10, dtype=tf.float32), (5, 2))
    fold = tf.constant([1, 0, 0, 0, 1], dtype=tf.int32)

    output = layer.call([x, fold], training=False)
    expected_output = tf.constant([[0.0, 1.0], [8.0, 9.0]])
    tf.debugging.assert_equal(output, expected_output)

    output = layer.call([x, fold], training=True)
    expected_output = tf.constant([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    tf.debugging.assert_equal(output, expected_output)

    config = layer.get_config()
    assert config["fold_index"] == 1


def test_CVMerge():
    layer = pytoolkit.tf.layers.CVMerge()
    fold = tf.constant([1, 0, 0, 0, 1], dtype=tf.int32)

    x0 = tf.constant([[5, 6], [7, 8], [9, 10]], dtype=tf.float32)
    x1 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    output = layer([x0, x1, fold], training=False)
    expected_output = tf.constant(
        [[1, 2], [5, 6], [7, 8], [9, 10], [3, 4]], dtype=tf.float32
    )
    tf.debugging.assert_equal(output, expected_output)

    x0 = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    x1 = tf.constant([[5, 6], [7, 8], [9, 10]], dtype=tf.float32)
    output = layer([x0, x1, fold], training=True)
    expected_output = tf.constant(
        [[1, 2], [5, 6], [7, 8], [9, 10], [3, 4]], dtype=tf.float32
    )
    tf.debugging.assert_equal(output, expected_output)


def test_CV():
    nfold = 3
    inputs = tf.reshape(tf.range(10, dtype=tf.float32), (5, 2))
    fold = tf.constant([2, 0, 0, 0, 2], dtype=tf.int32)

    input_features = tf.keras.Input((2,), name="features")
    input_fold = tf.keras.Input((), dtype="int32", name="fold")
    x = pytoolkit.tf.layers.CVMerge()(
        [
            pytoolkit.tf.layers.CVPick(fold_index)([input_features, input_fold]) * 0.0
            + fold_index
            for fold_index in range(nfold)
        ]
        + [input_fold]
    )
    model = tf.keras.models.Model([input_features, input_fold], x)

    output = model([inputs, fold], training=False)
    expected_output = tf.constant(
        [[2, 2], [0, 0], [0, 0], [0, 0], [2, 2]], dtype=tf.float32
    )
    tf.debugging.assert_equal(output, expected_output)

    output = model([inputs, fold], training=True)
    expected_output = tf.constant(
        [[0.5, 0.5], [1.5, 1.5], [1.5, 1.5], [1.5, 1.5], [0.5, 0.5]], dtype=tf.float32
    )
    tf.debugging.assert_equal(output, expected_output)
