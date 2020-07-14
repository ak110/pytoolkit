import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


def test_binary_iou():
    y_true = tf.constant(np.array([[0.0, 0.0, 1.0, 1.0]]))
    y_pred = tf.constant(np.array([[0.0, 0.9, 0.8, 0.0]]))
    metric = tk.metrics.binary_iou(y_true, y_pred, threshold=0.8)
    assert metric.numpy() == pytest.approx(1 / 3)


def test_bboxes_iou():
    y_true = tf.constant(
        [
            [
                [
                    [100, 100, 300, 300],
                    [100, 100, 300, 300],
                    [100, 100, 300, 300],
                    [0, 0, 0, 0],
                ]
            ]
        ],
        dtype=tf.float32,
    )
    y_pred = tf.constant(
        [
            [
                [
                    [100, 100, 300, 300],
                    [150, 150, 250, 250],
                    [0, 0, 0, 0],
                    [-1, -1, 0, 0],
                ],
            ],
        ],
        dtype=tf.float32,
    )
    assert y_true.shape.rank == 4
    assert y_pred.shape.rank == 4
    loss = tk.metrics.bboxes_iou(y_true, y_pred).numpy()
    assert loss.ndim == 3
    assert not np.isnan(loss).any()
    assert not np.isinf(loss).any()
    assert loss[..., 0] == 1.0
    assert (loss[..., 1:] < 1).all()
    assert (loss[..., 1:] >= 0).all()

    # scale
    loss_scaled = tk.metrics.bboxes_iou(y_true * 123, y_pred * 123).numpy()
    assert loss == pytest.approx(loss_scaled)
