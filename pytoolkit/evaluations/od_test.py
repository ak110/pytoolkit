import numpy as np

import pytoolkit as tk


def test_print_od_metrics():
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
    tk.evaluations.print_od_metrics(y_true, y_pred)
