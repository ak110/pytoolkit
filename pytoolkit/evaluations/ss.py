"""セマンティックセグメンテーションの評価。"""

import numpy as np

import pytoolkit as tk


def print_ss_metrics(y_true, y_pred, threshold=0.5, print_fn=None):
    """semantic segmentationの各種metricsを算出してprintする。"""
    print_fn = print_fn or tk.log.get(__name__)
    evals = evaluate_ss(y_true, y_pred, threshold)
    print_fn(f"IoU score:  {evals['iou_score']:.3f}")
    print_fn(f"Dice coef.: {evals['dice']:.3f}")
    print_fn(f"IoU mean:   {evals['iou_mean']:.3f}")
    print_fn(f"Acc empty:  {evals['acc_empty']:.3f}")
    return evals


def evaluate_ss(y_true, y_pred, threshold=0.5):
    """semantic segmentationの各種metricsを算出してdictで返す。

    Args:
        y_true (ndarray): ラベル (shape=(N, H, W) or (N, H, W, C))
        y_pred (ndarray): 推論結果 (shape=(N, H, W) or (N, H, W, C))

    Returns:
        dict: 各種metrics

    """
    mask_true = y_true >= threshold
    mask_pred = y_pred >= threshold

    obj = np.any(mask_true, axis=(1, 2, 3))
    empty = np.logical_not(obj)
    pred_empty = np.logical_not(np.any(mask_pred, axis=(1, 2, 3)))
    tn = np.logical_and(empty, pred_empty)

    # 塩コンペのスコア
    # https://www.kaggle.com/c/tgs-salt-identification-challenge/overview/evaluation
    inter = np.sum(np.logical_and(mask_true, mask_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(mask_true, mask_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)
    prec_list = []
    for th in np.arange(0.5, 1.0, 0.05):
        pred_obj = iou > th
        match = np.logical_and(obj, pred_obj) + tn
        prec_list.append(np.sum(match) / len(mask_true))
    iou_score = np.mean(prec_list)

    # mean Dice coefficient
    # https://www.kaggle.com/c/severstal-steel-defect-detection/overview/evaluation
    dice = 2 * inter / (mask_true.sum(axis=(1, 2, 3) + mask_pred.sum(axis=(1, 2, 3))))

    # 答えが空でないときのIoUの平均
    inter = np.sum(np.logical_and(mask_true, mask_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(mask_true, mask_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)
    iou_mean = np.mean(iou[obj])

    # 答えが空の場合の正解率
    pred_empty = np.logical_not(np.any(mask_pred, axis=(1, 2, 3)))
    acc_empty = np.sum(np.logical_and(empty, pred_empty)) / np.sum(empty)

    return {
        "iou_score": iou_score,
        "dice": dice,
        "iou_mean": iou_mean,
        "acc_empty": acc_empty,
    }
