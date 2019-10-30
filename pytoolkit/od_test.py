import numpy as np
import pytest

import pytoolkit as tk


def test_to_str():
    class_names = ["class00", "class01", "class02"]
    y = tk.od.ObjectsAnnotation(
        "path/to/dummy.jpg",
        100,
        100,
        np.array([1, 2]),
        np.array([[0.900, 0.900, 1.000, 1.000], [0.000, 0.000, 0.200, 0.200]]),
    )
    p = tk.od.ObjectsPrediction(
        np.array([1, 0, 2]),
        np.array([0.8, 0.1, 0.8]),
        np.array(
            [[0.900, 0.900, 1.000, 1.000], [0, 0, 1, 1], [0.000, 0.000, 0.200, 0.200]]
        ),
    )
    assert (
        y.to_str(class_names)
        == "(0, 0) [20 x 20]: class02\n(90, 90) [10 x 10]: class01"
    )
    assert (
        p.to_str(100, 100, class_names, 0.5)
        == "(0, 0) [20 x 20]: class02\n(90, 90) [10 x 10]: class01"
    )


def test_plot_objects(data_dir, check_dir):
    xml_path = data_dir / "od" / "Annotations" / "無題.xml"
    img_path = data_dir / "od" / "JPEGImages" / "無題.png"
    class_name_to_id = {"～": 0, "〇": 1}
    class_id_to_name = {v: k for k, v in class_name_to_id.items()}
    ann = tk.datasets.voc.load_annotation(data_dir / "od", xml_path, class_name_to_id)

    img = tk.od.plot_objects(img_path, ann.classes, None, ann.bboxes, None)
    tk.ndimage.save(check_dir / "plot_objects1.png", img)

    img = tk.od.plot_objects(img_path, ann.classes, None, ann.bboxes, class_id_to_name)
    tk.ndimage.save(check_dir / "plot_objects2.png", img)

    img = tk.od.plot_objects(
        img_path, ann.classes, np.array([0.5]), ann.bboxes, class_id_to_name
    )
    tk.ndimage.save(check_dir / "plot_objects3.png", img)


def test_iou():
    bboxes_a = np.array([[0, 0, 200, 200], [1000, 1000, 1001, 1001]])
    bboxes_b = np.array([[100, 100, 300, 300]])
    iou = tk.od.compute_iou(bboxes_a, bboxes_b)
    assert iou.shape == (2, 1)
    assert iou[0, 0] == pytest.approx(100 * 100 / (200 * 200 * 2 - 100 * 100))
    assert iou[1, 0] == 0


def test_is_in_box():
    boxes_a = np.array([[100, 100, 300, 300]])
    boxes_b = np.array(
        [
            [150, 150, 250, 250],
            [100, 100, 300, 300],
            [50, 50, 350, 350],
            [150, 150, 350, 350],
        ]
    )
    is_in = tk.od.is_in_box(boxes_a, boxes_b)
    assert is_in.shape == (len(boxes_a), len(boxes_b))
    assert (is_in == [[False, True, True, False]]).all()


def test_od_accuracy():
    y_true = np.tile(
        np.array(
            [
                tk.od.ObjectsAnnotation(
                    path=".",
                    width=100,
                    height=100,
                    classes=[0, 1],
                    bboxes=[[0.00, 0.00, 0.05, 0.05], [0.25, 0.25, 0.75, 0.75]],
                )
            ]
        ),
        6,
    )
    y_pred = np.array(
        [
            # 一致
            tk.od.ObjectsPrediction(
                classes=[1, 0],
                confs=[1, 1],
                bboxes=[[0.25, 0.25, 0.75, 0.75], [0.00, 0.00, 0.05, 0.05]],
            ),
            # conf低
            tk.od.ObjectsPrediction(
                classes=[1, 0, 0],
                confs=[1, 0, 1],
                bboxes=[
                    [0.25, 0.25, 0.75, 0.75],
                    [0.00, 0.00, 0.05, 0.05],
                    [0.00, 0.00, 0.05, 0.05],
                ],
            ),
            # クラス違い
            tk.od.ObjectsPrediction(
                classes=[1, 1],
                confs=[1, 1],
                bboxes=[[0.25, 0.25, 0.75, 0.75], [0.00, 0.00, 0.05, 0.05]],
            ),
            # 重複
            tk.od.ObjectsPrediction(
                classes=[1, 0, 0],
                confs=[1, 1, 1],
                bboxes=[
                    [0.25, 0.25, 0.75, 0.75],
                    [0.00, 0.00, 0.05, 0.05],
                    [0.00, 0.00, 0.05, 0.05],
                ],
            ),
            # 不足
            tk.od.ObjectsPrediction(
                classes=[1], confs=[1], bboxes=[[0.25, 0.25, 0.75, 0.75]]
            ),
            # IoU低
            tk.od.ObjectsPrediction(
                classes=[1, 0],
                confs=[1, 1],
                bboxes=[[0.25, 0.25, 0.75, 0.75], [0.90, 0.90, 0.95, 0.95]],
            ),
        ]
    )
    is_match_expected = [True, True, False, False, False, False]
    for yt, yp, m in zip(y_true, y_pred, is_match_expected):
        assert yp.is_match(yt.classes, yt.bboxes, conf_threshold=0.5) == m
    assert tk.od.od_accuracy(y_true, y_pred, conf_threshold=0.5) == pytest.approx(2 / 6)


def test_confusion_matrix():
    y_true = np.array([])
    y_pred = np.array([])
    cm_actual = tk.od.confusion_matrix(y_true, y_pred, num_classes=3)
    assert (cm_actual == np.zeros((4, 4), dtype=np.int32)).all()

    y_true = np.array(
        [
            tk.od.ObjectsAnnotation(
                path=".",
                width=100,
                height=100,
                classes=[1],
                bboxes=[[0.25, 0.25, 0.75, 0.75]],
            )
        ]
    )
    y_pred = np.array(
        [
            tk.od.ObjectsPrediction(
                classes=[0, 2, 1, 1, 2],
                confs=[0, 1, 1, 1, 1],
                bboxes=[
                    [0.25, 0.25, 0.75, 0.75],  # conf低
                    [0.25, 0.25, 0.75, 0.75],  # クラス違い
                    [0.25, 0.25, 0.75, 0.75],  # 検知
                    [0.25, 0.25, 0.75, 0.75],  # 重複
                    [0.95, 0.95, 0.99, 0.99],  # IoU低
                ],
            )
        ]
    )
    cm_actual = tk.od.confusion_matrix(
        y_true, y_pred, conf_threshold=0.5, num_classes=3
    )
    cm_expected = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 2, 0]], dtype=np.int32
    )
    assert (cm_actual == cm_expected).all()
