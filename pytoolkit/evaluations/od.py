"""物体検出の評価。"""
from __future__ import annotations

import typing
import numpy as np

import pytoolkit as tk


def print_od_metrics(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
    print_fn: typing.Callable[[str], None] = None,
) -> tk.evaluations.EvalsType:
    """物体検出の各種metricsを算出してprintする。

    Args:
        y_true: ラベル
        y_pred: 推論結果

    Returns:
        各種metrics

    """
    evals = evaluate_od(y_true, y_pred)
    print_fn = print_fn or tk.log.get(__name__).info
    print_fn(f"BOX AP:    {evals['map/iou=0.50:0.95/area=all/max_dets=100']:.3f}")
    print_fn(f"AP50:      {evals['map/iou=0.50/area=all/max_dets=100']:.3f}")
    print_fn(f"AP75:      {evals['map/iou=0.75/area=all/max_dets=100']:.3f}")
    print_fn(f"APS:       {evals['map/iou=0.50:0.95/area=small/max_dets=100']:.3f}")
    print_fn(f"APM:       {evals['map/iou=0.50:0.95/area=medium/max_dets=100']:.3f}")
    print_fn(f"APL:       {evals['map/iou=0.50:0.95/area=large/max_dets=100']:.3f}")
    print_fn(f"VOC07 MAP: {evals['voc07_map']:.3f}")
    return evals


def evaluate_od(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
    detail: bool = False,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> tk.evaluations.EvalsType:
    """物体検出の各種metricsを算出してdictで返す。

    Args:
        y_true: ラベル
        y_pred: 推論結果
        detail: 全情報を返すならTrue。(既定値では一部の指標のみ返す)
        conf_threshold: 確信度の閾値 (accなどに影響)
        iou_threshold: 一致扱いする最低IoU (accなどに影響)

    Returns:
        各種metrics

        - "iou_score": IoUスコア (塩コンペのスコア)
        - "dice": ダイス係数
        - "fg_iou": 答えが空でないときのIoUの平均
        - "bg_acc": 答えが空の場合の正解率
        - "acc": Pixel Accuracy

    ChainerCVを利用。
    https://chainercv.readthedocs.io/en/stable/reference/evaluations.html?highlight=eval_detection_coco#chainercv.evaluations.eval_detection_coco
    https://chainercv.readthedocs.io/en/stable/reference/evaluations.html?highlight=eval_detection_coco#eval-detection-voc

    Returns:
        - "map/iou=0.50:0.95/area=all/max_dets=100"
        - "map/iou=0.50/area=all/max_dets=100"
        - "map/iou=0.75/area=all/max_dets=100"
        - …などなどcoco関連
        - "voc_ap"
        - "voc_map"
        - "voc07_ap"
        - "voc07_map"

    """
    import chainercv

    gt_classes_list = np.array([y.classes for y in y_true])
    gt_bboxes_list = np.array([y.real_bboxes for y in y_true])
    gt_areas_list = np.array([y.areas for y in y_true])
    gt_crowdeds_list = np.array([y.crowdeds for y in y_true])
    gt_difficults_list = np.array([y.difficults for y in y_true])
    pred_classes_list = np.array([p.classes for p in y_pred])
    pred_confs_list = np.array([p.confs for p in y_pred])
    pred_bboxes_list = np.array(
        [p.get_real_bboxes(y.width, y.height) for (p, y) in zip(y_pred, y_true)]
    )
    with np.errstate(all="warn"):
        evals = chainercv.evaluations.eval_detection_coco(
            pred_bboxes_list,
            pred_classes_list,
            pred_confs_list,
            gt_bboxes_list,
            gt_classes_list,
            gt_areas_list,
            gt_crowdeds_list,
        )
        voc_evals = chainercv.evaluations.eval_detection_voc(
            pred_bboxes_list,
            pred_classes_list,
            pred_confs_list,
            gt_bboxes_list,
            gt_classes_list,
            gt_difficults_list,
            use_07_metric=True,
        )
        evals["voc07_ap"] = voc_evals["ap"]
        evals["voc07_map"] = voc_evals["map"]

        if not detail:
            summary_keys = [
                "map/iou=0.50:0.95/area=all/max_dets=100",
                "map/iou=0.50/area=all/max_dets=100",
                "map/iou=0.75/area=all/max_dets=100",
                "map/iou=0.50:0.95/area=small/max_dets=100",
                "map/iou=0.50:0.95/area=medium/max_dets=100",
                "map/iou=0.50:0.95/area=large/max_dets=100",
                "voc07_map",
                "ar/iou=0.50:0.95/area=all/max_dets=100",
                "voc07_ap",
            ]
            evals = {k: evals[k] for k in summary_keys}

        # 独自指標
        acc = tk.od.od_accuracy(
            y_true, y_pred, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        prec, rec, fscores, _ = tk.od.compute_scores(
            y_true, y_pred, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        cm = tk.od.confusion_matrix(
            y_true, y_pred, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        evals["acc"] = acc
        evals["prec"] = prec
        evals["rec"] = rec
        evals["F"] = fscores
        evals["cm"] = cm

        return evals
