"""分類の評価。"""
from __future__ import annotations

import typing

import numpy as np
import sklearn.metrics

import pytoolkit as tk


def print_classification(
    y_true: np.ndarray,
    proba_pred: np.ndarray,
    average: str = "macro",
    print_fn: typing.Callable[[str], None] = None,
) -> tk.evaluations.EvalsType:
    """分類の指標色々を表示する。"""
    try:
        evals = evaluate_classification(y_true, proba_pred, average)
        print_fn = print_fn or tk.log.get(__name__).info
        print_fn(f"Accuracy:  {evals['acc']:.3f} (Error: {evals['error']:.3f})")
        print_fn(f"F1-score:  {evals['f1']:.3f}")
        print_fn(f"AUC:       {evals['auc']:.3f}")
        print_fn(f"AP:        {evals['ap']:.3f}")
        print_fn(f"Precision: {evals['prec']:.3f}")
        print_fn(f"Recall:    {evals['rec']:.3f}")
        print_fn(f"Logloss:   {evals['logloss']:.3f}")
        return evals
    except Exception:
        tk.log.get(__name__).warning("Error: print_classification", exc_info=True)
        return {}


def evaluate_classification(
    y_true: np.ndarray, proba_pred: np.ndarray, average: str = "macro"
) -> tk.evaluations.EvalsType:
    """分類の評価。"""
    with np.errstate(all="warn"):
        true_type = sklearn.utils.multiclass.type_of_target(y_true)
        pred_type = sklearn.utils.multiclass.type_of_target(proba_pred)
        if true_type == "binary":  # binary
            assert pred_type in ("binary", "continuous", "continuous-multioutput")
            if pred_type == "continuous-multioutput":
                assert proba_pred.shape == (
                    len(proba_pred),
                    2,
                ), f"Shape error: {proba_pred.shape}"
                proba_pred = proba_pred[:, 1]
            y_pred = (np.asarray(proba_pred) >= 0.5).astype(np.int32)
            acc = sklearn.metrics.accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = sklearn.metrics.precision_recall_fscore_support(
                y_true, y_pred
            )
            mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
            auc = sklearn.metrics.roc_auc_score(y_true, proba_pred)
            ap = sklearn.metrics.average_precision_score(y_true, proba_pred)
            logloss = sklearn.metrics.log_loss(y_true, proba_pred)
            return {
                "acc": acc,
                "error": 1 - acc,
                "f1": f1,
                "auc": auc,
                "ap": ap,
                "prec": prec,
                "rec": rec,
                "mcc": mcc,
                "logloss": logloss,
            }
        else:  # multiclass
            assert true_type == "multiclass"
            assert pred_type == "continuous-multioutput"
            num_classes = np.max(y_true) + 1
            labels = list(range(num_classes))
            ohe_true = tk.ml.to_categorical(num_classes)(np.asarray(y_true))
            y_pred = np.argmax(proba_pred, axis=-1)
            acc = sklearn.metrics.accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = sklearn.metrics.precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=average
            )
            mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
            auc = sklearn.metrics.roc_auc_score(ohe_true, proba_pred, average=average)
            ap = sklearn.metrics.average_precision_score(
                ohe_true, proba_pred, average=average
            )
            logloss = sklearn.metrics.log_loss(ohe_true, proba_pred)
            return {
                "acc": acc,
                "error": 1 - acc,
                "f1": f1,
                "auc": auc,
                "ap": ap,
                "prec": prec,
                "rec": rec,
                "mcc": mcc,
                "logloss": logloss,
            }
