import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_iou():
    boxes = np.array([
        [0, 0, 200, 200],
        [1000, 1000, 1001, 1001],
    ])
    iou = tk.ml.iou(boxes, np.array([100, 100, 300, 300]))
    assert len(iou) == 2
    assert iou[0] == pytest.approx(100 * 100 / (200 * 200 * 2 - 100 * 100))
    assert iou[1] == 0


def test_plot_cm(tmpdir):
    filepath = str(tmpdir.join('confusion_matrix.png'))
    cm = [
        [5, 0, 0],
        [0, 3, 2],
        [0, 2, 3],
    ]
    tk.ml.plot_cm(cm, filepath)
    assert pathlib.Path(filepath).is_file()  # とりあえずファイルの存在チェックだけ。。
