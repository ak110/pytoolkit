import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_compute_map():
    gt_classes_list = [np.array([1, 2])]
    gt_bboxes_list = [np.array([
        [0, 0, 200, 200],
        [900, 900, 1000, 1000],
    ])]
    gt_difficults_list = [np.array([False, False])]
    pred_class_list1 = [np.array([9, 9])]
    pred_class_list2 = [np.array([9, 1])]
    pred_class_list3 = [np.array([2, 9])]
    pred_class_list4 = [np.array([2, 1])]
    pred_bboxes_list = [np.array([
        [901, 901, 1000, 1000],
        [1, 1, 199, 199],
    ])]
    assert tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_class_list1, pred_bboxes_list) == 0
    assert tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_class_list2, pred_bboxes_list) == 0.25
    assert tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_class_list3, pred_bboxes_list) == 0.5
    assert tk.ml.compute_map(gt_classes_list, gt_bboxes_list, gt_difficults_list, pred_class_list4, pred_bboxes_list) == 1


def test_iou():
    bboxes_a = np.array([
        [0, 0, 200, 200],
        [1000, 1000, 1001, 1001],
    ])
    bboxes_b = np.array([
        [100, 100, 300, 300]
    ])
    iou = tk.ml.iou(bboxes_a, bboxes_b)
    assert iou.shape == (2, 1)
    assert iou[0][0] == pytest.approx(100 * 100 / (200 * 200 * 2 - 100 * 100))
    assert iou[1][0] == 0


def test_plot_cm(tmpdir):
    filepath = str(tmpdir.join('confusion_matrix.png'))
    cm = [
        [5, 0, 0],
        [0, 3, 2],
        [0, 2, 3],
    ]
    tk.ml.plot_cm(cm, filepath)
    assert pathlib.Path(filepath).is_file()  # とりあえずファイルの存在チェックだけ。。
