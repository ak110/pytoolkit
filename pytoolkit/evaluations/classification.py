"""分類の評価。"""
import numpy as np
import sklearn.metrics

import pytoolkit as tk


def print_classification_metrics(y_true, proba_pred, average="macro", print_fn=None):
    """分類の指標色々を表示する。"""
    try:
        evals = evaluate_classification(y_true, proba_pred, average)
        print_fn = print_fn or tk.log.get(__name__).info
        if evals["type"] == "binary":  # binary
            print_fn(f"Accuracy:  {evals['acc']:.3f} (Error: {1 - evals['acc']:.3f})")
            print_fn(f"F1-score:  {evals['f1']:.3f}")
            print_fn(f"AUC:       {evals['auc']:.3f}")
            print_fn(f"AP:        {evals['ap']:.3f}")
            print_fn(f"Precision: {evals['prec']:.3f}")
            print_fn(f"Recall:    {evals['rec']:.3f}")
            print_fn(f"Logloss:   {evals['logloss']:.3f}")
        else:  # multiclass
            print_fn(f"Accuracy:   {evals['acc']:.3f} (Error: {1 - evals['acc']:.3f})")
            print_fn(f"F1-{average:5s}:   {evals['f1']:.3f}")
            print_fn(f"AUC-{average:5s}:  {evals['auc']:.3f}")
            print_fn(f"AP-{average:5s}:   {evals['ap']:.3f}")
            print_fn(f"Prec-{average:5s}: {evals['prec']:.3f}")
            print_fn(f"Rec-{average:5s}:  {evals['rec']:.3f}")
            print_fn(f"Logloss:    {evals['logloss']:.3f}")
        return evals
    except BaseException:
        tk.log.get(__name__).warning(
            "Error: print_classification_metrics", exc_info=True
        )


def evaluate_classification(y_true, proba_pred, average="macro"):
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
        auc = sklearn.metrics.roc_auc_score(y_true, proba_pred)
        ap = sklearn.metrics.average_precision_score(y_true, proba_pred)
        logloss = sklearn.metrics.log_loss(y_true, proba_pred)
        return {
            "type": "binary",
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "ap": ap,
            "prec": prec,
            "rec": rec,
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
        auc = sklearn.metrics.roc_auc_score(ohe_true, proba_pred, average=average)
        ap = sklearn.metrics.average_precision_score(
            ohe_true, proba_pred, average=average
        )
        logloss = sklearn.metrics.log_loss(ohe_true, proba_pred)
        return {
            "type": "multiclass",
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "ap": ap,
            "prec": prec,
            "rec": rec,
            "logloss": logloss,
        }
