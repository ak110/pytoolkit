import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_plot_objects():
    base_dir = pathlib.Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    xml_path = data_dir / 'VOC2007_000001.xml'
    img_path = data_dir / 'VOC2007_000001.jpg'
    ann = tk.data.voc.load_annotation(data_dir, xml_path)

    img = tk.ml.plot_objects(img_path, ann.classes, None, ann.bboxes, None)
    tk.ndimage.save(base_dir.parent / '___check' / 'plot_objects1.png', img)

    img = tk.ml.plot_objects(img_path, ann.classes, None, ann.bboxes, tk.data.voc.CLASS_NAMES)
    tk.ndimage.save(base_dir.parent / '___check' / 'plot_objects2.png', img)

    img = tk.ml.plot_objects(img_path, ann.classes, np.array([0.5, 0.5]), ann.bboxes, tk.data.voc.CLASS_NAMES)
    tk.ndimage.save(base_dir.parent / '___check' / 'plot_objects3.png', img)


def test_compute_map():
    gt = [
        tk.ml.ObjectsAnnotation(
            'path/to/dummy.jpg', 2000, 2000,
            np.array([1, 2]),
            np.array([
                [0.000, 0.000, 0.200, 0.200],
                [0.900, 0.900, 1.000, 1.000],
            ]),
            np.array([False, False]))
    ]
    preds = [
        tk.ml.ObjectsPrediction(
            np.array([9, 9, 3, 3, 3, 3]),
            np.array([1, 1, 1, 1, 1, 1]),
            np.array([
                [0.901, 0.901, 1.000, 1.000],
                [0.001, 0.001, 0.199, 0.199],
                [0.333, 0.333, 0.334, 0.334],  # error
                [0.333, 0.333, 0.334, 0.334],  # error
                [0.333, 0.333, 0.334, 0.334],  # error
                [0.333, 0.333, 0.334, 0.334],  # error
            ])),
    ]
    assert tk.ml.compute_map(gt, preds) == 0

    preds[0].classes = np.array([9, 1, 3, 3, 3, 3])
    assert tk.ml.compute_map(gt, preds) == 0.5

    preds[0].classes = np.array([2, 9, 3, 3, 3, 3])
    assert tk.ml.compute_map(gt, preds) == 0.5

    preds[0].classes = np.array([2, 1, 3, 3, 3, 3])
    assert tk.ml.compute_map(gt, preds) == 1


def test_iou():
    bboxes_a = np.array([
        [0, 0, 200, 200],
        [1000, 1000, 1001, 1001],
    ])
    bboxes_b = np.array([
        [100, 100, 300, 300]
    ])
    iou = tk.ml.compute_iou(bboxes_a, bboxes_b)
    assert iou.shape == (2, 1)
    assert iou[0, 0] == pytest.approx(100 * 100 / (200 * 200 * 2 - 100 * 100))
    assert iou[1, 0] == 0


def test_size_based_iou():
    bboxes_a = np.array([[10, 15]])
    bboxes_b = np.array([[15, 10]])
    iou = tk.ml.compute_size_based_iou(bboxes_a, bboxes_b)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == pytest.approx(0.5)


def test_is_in_box():
    boxes_a = np.array([
        [100, 100, 300, 300]
    ])
    boxes_b = np.array([
        [150, 150, 250, 250],
        [100, 100, 300, 300],
        [50, 50, 350, 350],
        [150, 150, 350, 350],
    ])
    is_in = tk.ml.is_in_box(boxes_a, boxes_b)
    assert is_in.shape == (len(boxes_a), len(boxes_b))
    assert (is_in == [[False, True, True, False]]).all()


def test_top_k_accuracy():
    y_true = np.array([1, 1, 1])
    proba_pred = np.array([
        [0.2, 0.1, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.3, 0.2],
    ])
    assert tk.ml.top_k_accuracy(y_true, proba_pred, k=2) == pytest.approx(2 / 3)


def test_non_maximum_suppression():
    boxes = np.array([
        [200, 0, 200, 200],  # empty
        [0, 201, 200, 200],  # empty
        [0, 0, 200, 200],  # col
        [0, 0, 201, 201],  # col
        [0, 0, 202, 202],  # col
        [1000, 1000, 1001, 1001],
    ])
    scores = np.array([
        0.9,
        0.9,
        0.7,
        0.8,
        0.6,
        0.85,
    ])
    idx = tk.ml.non_maximum_suppression(boxes, scores, 200, 0.45)
    assert (idx == np.array([5, 3])).all()

    idx = tk.ml.non_maximum_suppression(np.array([[0, 0, 200, 200]]), np.array([0.2]), 200, 0.45)
    assert (idx == np.array([0])).all()

    idx = tk.ml.non_maximum_suppression(np.array([[200, 0, 200, 200]]), np.array([0.1]), 200, 0.45)
    assert idx.shape == (0,)


def test_print_classification_metrics_multi():
    y_true = [0, 1, 1, 1, 2]
    prob_pred = [
        [0.75, 0.00, 0.25],
        [0.25, 0.75, 0.00],
        [0.25, 0.75, 0.00],
        [0.25, 0.00, 0.75],
        [0.25, 0.75, 0.00],
    ]
    tk.ml.print_classification_metrics(y_true, prob_pred)


def test_print_classification_metrics_binary():
    y_true = [0, 1, 1, 0]
    prob_pred = [0.25, 0.25, 0.75, 0.25]
    tk.ml.print_classification_metrics(y_true, prob_pred)


def test_plot_cm():
    filepath = pathlib.Path(__file__).resolve().parent.parent / '___check' / 'plot_cm.png'
    cm = [
        [5, 0, 0],
        [0, 3, 2],
        [0, 2, 3],
    ]
    tk.ml.plot_cm(cm, filepath)
