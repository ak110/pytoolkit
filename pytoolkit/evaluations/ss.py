"""セマンティックセグメンテーションの評価。"""

import numpy as np

import pytoolkit as tk


def print_ss_metrics(y_true, y_pred, threshold=0.5, print_fn=None):
    """semantic segmentationの各種metricsを算出してprintする。

    Args:
        y_true (array-like): ラベル (shape=(N, H, W) or (N, H, W, C))
        y_pred (array-like): 推論結果 (shape=(N, H, W) or (N, H, W, C))

    Returns:
        dict: 各種metrics

    """
    print_fn = print_fn or tk.log.get(__name__)
    evals = evaluate_ss(y_true, y_pred, threshold)
    print_fn(f"IoU:        {evals['iou']:.3f}")
    if np.ndim(evals["iou"]) >= 1:
        print_fn(f"mIoU:       {evals['miou']:.3f}")
    print_fn(f"IoU score:  {evals['iou_score']:.3f}")
    print_fn(f"Dice coef.: {evals['dice']:.3f}")
    print_fn(f"IoU mean:   {evals['iou_mean']:.3f}")
    print_fn(f"Acc empty:  {evals['acc_empty']:.3f}")
    return evals


def evaluate_ss(y_true, y_pred, threshold=0.5):
    """semantic segmentationの各種metricsを算出してdictで返す。

    y_true, y_predはgeneratorも可。(メモリ不足にならないように)

    Args:
        y_true (array-like): ラベル (shape=(N, H, W) or (N, H, W, C))
        y_pred (array-like): 推論結果 (shape=(N, H, W) or (N, H, W, C))

    Returns:
        dict: 各種metrics

        "iou_score": IoUスコア (塩コンペのスコア)
        "dice": ダイス係数
        "iou_mean": 答えが空でないときのIoUの平均
        "acc_empty": 答えが空の場合の正解率

    References:
        - <https://www.kaggle.com/c/tgs-salt-identification-challenge/overview/evaluation>
        - <https://www.kaggle.com/c/severstal-steel-defect-detection/overview/evaluation>

    """

    def process_per_image(yt, yp):
        if np.ndim(yt) == 3:
            yt = np.expand_dims(yt, axis=-1)
        if np.ndim(yp) == 3:
            yp = np.expand_dims(yp, axis=-1)
        assert np.ndim(yt) == 4
        assert np.ndim(yp) == 4
        p_true = yt >= threshold
        p_pred = yp >= threshold
        n_true = np.logical_not(p_true)
        n_pred = np.logical_not(p_pred)
        tp = np.sum(np.logical_and(p_true, p_pred), axis=(0, 1))
        fp = np.sum(np.logical_and(n_true, p_pred), axis=(0, 1))
        tn = np.sum(np.logical_and(n_true, n_pred), axis=(0, 1))
        fn = np.sum(np.logical_and(p_true, n_pred), axis=(0, 1))
        return p_true, p_pred, tp, fp, tn, fn

    r = [process_per_image(yt, yp) for yt, yp in zip(y_true, y_pred)]
    p_true, p_pred, tp, fp, tn, fn = zip(*r)
    p_true = np.array(p_true)
    p_pred = np.array(p_pred)
    tp = np.array(tp)
    fp = np.array(fp)
    tn = np.array(tn)
    fn = np.array(fn)

    sample_iou = tp.sum(axis=1) / (tp.sum(axis=1) + fp.sum(axis=1) + tn.sum(axis=1))
    class_iou = tp.sum(axis=0) / (tp.sum(axis=0) + fp.sum(axis=0) + tn.sum(axis=0))
    class_dice = 2 * tp.sum(axis=0) / (p_true.sum(axis=0) + p_pred.sum(axis=0))

    # 塩コンペのスコア
    prec_list = []
    for th in np.arange(0.5, 1.0, 0.05):
        pred_obj = sample_iou > th
        match = np.logical_and(p_true, pred_obj) + tn
        prec_list.append(np.mean(match))
    iou_score = np.mean(prec_list)

    return {
        "iou": class_iou,
        "miou": np.mean(class_iou) if np.ndim(class_iou) >= 1 else class_iou,
        "iou_score": iou_score,
        "dice": np.mean(class_dice),
        "iou_mean": np.mean(sample_iou[p_true]),
        "acc_empty": np.mean(tn[np.logical_not(p_true)]),
    }
