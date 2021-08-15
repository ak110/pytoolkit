import numpy as np

import pytoolkit as tk


def test_print_od():
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

    evals = tk.evaluations.print_od(y_true, y_pred)
    np.testing.assert_allclose(
        evals["map/iou=0.50:0.95/area=all/max_dets=100"], 0.717115, rtol=1e-5
    )
    np.testing.assert_allclose(
        evals["map/iou=0.50/area=all/max_dets=100"], 0.717115, rtol=1e-5
    )
    np.testing.assert_allclose(
        evals["map/iou=0.75/area=all/max_dets=100"], 0.717115, rtol=1e-5
    )
    np.testing.assert_allclose(
        evals["map/iou=0.50:0.95/area=small/max_dets=100"], 0.504951, rtol=1e-5
    )
    np.testing.assert_allclose(evals["map/iou=0.50:0.95/area=medium/max_dets=100"], 1.0)
    np.testing.assert_allclose(
        evals["map/iou=0.50:0.95/area=large/max_dets=100"], np.nan
    )
