"""物体検出の評価。"""
from __future__ import annotations

import io
import logging
import sys
import typing

import numpy as np

import pytoolkit as tk

logger = logging.getLogger(__name__)


def print_od(
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
    print_fn = print_fn or logger.info
    print_fn(f"BOX AP:    {evals['map/iou=0.50:0.95/area=all/max_dets=100']:.3f}")
    print_fn(f"AP50:      {evals['map/iou=0.50/area=all/max_dets=100']:.3f}")
    print_fn(f"AP75:      {evals['map/iou=0.75/area=all/max_dets=100']:.3f}")
    print_fn(f"APS:       {evals['map/iou=0.50:0.95/area=small/max_dets=100']:.3f}")
    print_fn(f"APM:       {evals['map/iou=0.50:0.95/area=medium/max_dets=100']:.3f}")
    print_fn(f"APL:       {evals['map/iou=0.50:0.95/area=large/max_dets=100']:.3f}")
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
        - "map/iou=0.50:0.95/area=all/max_dets=100"
        - "map/iou=0.50/area=all/max_dets=100"
        - "map/iou=0.75/area=all/max_dets=100"

    """
    with np.errstate(all="warn"):
        # pycocotoolsを使った評価
        evals = _evaluate_coco(y_true, y_pred)
        if not detail:
            summary_keys = [
                "map/iou=0.50:0.95/area=all/max_dets=100",
                "map/iou=0.50/area=all/max_dets=100",
                "map/iou=0.75/area=all/max_dets=100",
                "map/iou=0.50:0.95/area=small/max_dets=100",
                "map/iou=0.50:0.95/area=medium/max_dets=100",
                "map/iou=0.50:0.95/area=large/max_dets=100",
                "ap/iou=0.50:0.95/area=all/max_dets=100",
                "ar/iou=0.50:0.95/area=all/max_dets=100",
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
        evals["prec-macro"] = np.nanmean(prec)
        evals["rec-macro"] = np.nanmean(rec)
        evals["F1-macro"] = np.nanmean(fscores)
        evals["prec"] = prec
        evals["rec"] = rec
        evals["F1"] = fscores
        evals["cm"] = cm

        return evals


def _evaluate_coco(
    y_true: typing.Sequence[tk.od.ObjectsAnnotation],
    y_pred: typing.Sequence[tk.od.ObjectsPrediction],
):
    """pycocotoolsを使った評価。"""
    import pycocotools

    img_ids = [{"id": i + 1} for i in range(len(y_true))]

    all_classes = np.sort(
        np.unique(
            np.concatenate([y.classes for y in y_true] + [y.classes for y in y_pred])
        )
    )
    class_ids = [{"id": c} for c in all_classes]

    gt_anns: typing.List[typing.Dict] = []
    pr_anns: typing.List[typing.Dict] = []
    for i, (y1, y2) in enumerate(zip(y_true, y_pred)):
        img_id = i + 1
        areas = y1.areas if y1.areas is not None else [None] * y1.num_objects
        crowdeds = y1.crowdeds if y1.crowdeds is not None else [None] * y1.num_objects
        for bbox, class_id, area, crowded in zip(
            y1.real_bboxes, y1.classes, areas, crowdeds
        ):
            gt_anns.append(
                _create_ann(
                    bbox=bbox,
                    class_id=class_id,
                    conf=None,
                    area=area,
                    crowded=crowded,
                    img_id=img_id,
                    ann_id=len(gt_anns) + 1,
                )
            )
        for bbox, class_id, conf in zip(
            y2.get_real_bboxes(y1.width, y1.height), y2.classes, y2.confs
        ):
            pr_anns.append(
                _create_ann(
                    bbox=bbox,
                    class_id=class_id,
                    conf=conf,
                    area=None,
                    crowded=0,
                    img_id=img_id,
                    ann_id=len(pr_anns) + 1,
                )
            )

    gt = pycocotools.coco.COCO()
    pr = pycocotools.coco.COCO()
    gt.dataset["images"] = img_ids
    pr.dataset["images"] = img_ids
    pr.dataset["categories"] = class_ids
    gt.dataset["categories"] = class_ids
    gt.dataset["annotations"] = gt_anns
    pr.dataset["annotations"] = pr_anns
    old_stdout, sys.stdout = sys.stdout, io.StringIO()  # stdout抑止
    try:
        pr.createIndex()
        gt.createIndex()
        coco_eval = pycocotools.cocoeval.COCOeval(gt, pr, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
    finally:
        sys.stdout = old_stdout

    evals = {}
    for type_ in ["ap", "ar"]:
        for iou, area, max_dets in [
            (None, "all", 100),
            (0.50, "all", 100),
            (0.75, "all", 100),
            (None, "small", 100),
            (None, "medium", 100),
            (None, "large", 100),
        ]:
            iou_str = f"{iou:.2f}" if iou is not None else "0.50:0.95"
            key = f"{type_}/iou={iou_str}/area={area}/max_dets={max_dets}"

            if type_ == "ap":
                values = coco_eval.eval["precision"].copy()
            else:
                values = coco_eval.eval["recall"].copy()
            assert values.shape[0] == len(coco_eval.params.iouThrs)
            assert values.shape[-2] == len(coco_eval.params.areaRngLbl)
            assert values.shape[-1] == len(coco_eval.params.maxDets)
            if iou is not None:
                values = values[iou == coco_eval.params.iouThrs]
            ai = coco_eval.params.areaRngLbl.index(area)
            mi = coco_eval.params.maxDets.index(max_dets)
            values = values[..., ai, mi]

            values[values == -1] = np.nan
            values = values.reshape((-1, values.shape[-1]))

            cls_values = [
                np.nanmean(values[:, c], axis=0)
                if not np.all(np.isnan(values[:, c]))
                else np.nan
                for c in all_classes
            ]
            evals[key] = cls_values
            evals[f"m{key}"] = (
                np.nanmean(cls_values) if not np.all(np.isnan(cls_values)) else np.nan
            )

    return evals


def _create_ann(bbox, class_id, conf, area, crowded, img_id, ann_id):
    size = bbox[2:] - bbox[:2]
    if area is None:
        area = size[0] * size[1]
    if crowded is None:
        crowded = 0
    ann = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": class_id,
        "bbox": [
            np.round(bbox[0], 2),
            np.round(bbox[1], 2),
            np.round(size[0], 2),
            np.round(size[1], 2),
        ],
        "area": area,
        "iscrowd": crowded,
    }
    if conf is not None:
        ann["score"] = conf
    return ann
