"""機械学習関連。"""
import pathlib

import joblib
import numpy as np
import sklearn.base
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils

import pytoolkit as tk

from . import log as tk_log


def listup_classification(dirpath, class_names=None, use_tqdm=True, check_image=False):
    """画像分類でよくある、クラス名ディレクトリの列挙。クラス名の配列, X, yを返す。

    Args:
        class_names: クラス名の配列
        use_tqdm: tqdmを使用するか否か
        check_image: 画像として読み込みチェックを行い、読み込み可能なファイルのみ返すか否か (遅いので注意)

    Returns:
        tuple: class_names, X, y

    """
    dirpath = pathlib.Path(dirpath)

    # クラス名
    if class_names is None:

        def _is_valid_classdir(p):
            if not p.is_dir():
                return False
            if p.name.lower() in (".svn", ".git"):  # 気休め程度に無視パターン
                return False
            return True

        class_names = list(
            sorted([p.name for p in dirpath.iterdir() if _is_valid_classdir(p)])
        )

    # 各クラスのデータを列挙
    X, y, errors = [], [], []
    for class_id, class_name in enumerate(
        tk.utils.tqdm(class_names, desc="listup", disable=not use_tqdm)
    ):
        class_dir = dirpath / class_name
        if class_dir.is_dir():
            t, err = _listup_files(
                class_dir, recurse=False, use_tqdm=False, check_image=check_image
            )
            X.extend(t)
            y.extend([class_id] * len(t))
            errors.extend(err)
    assert len(X) == len(y)
    for e in errors:
        print(e)
    return class_names, np.array(X), np.array(y)


def listup_files(dirpath, recurse=False, use_tqdm=True, check_image=False):
    """ファイルの列挙。

    Args:
        recurse: 再帰的に配下もリストアップするか否か
        use_tqdm: tqdmを使用するか否か
        check_image: 画像として読み込みチェックを行い、読み込み可能なファイルのみ返すか否か (遅いので注意)

    """
    result, errors = _listup_files(dirpath, recurse, use_tqdm, check_image)
    for e in errors:
        print(e)
    return np.array(result)


def _listup_files(dirpath, recurse, use_tqdm, check_image):
    """ファイルの列挙。"""
    errors = []

    dirpath = pathlib.Path(dirpath)
    if recurse:
        it = dirpath.rglob("*")
    else:
        it = dirpath.iterdir()

    def _is_valid_file(p):
        if not p.is_file():
            return False
        if p.name.lower() == "thumbs.db":
            return False
        if check_image:
            try:
                tk.ndimage.load(p)
            except BaseException:
                errors.append(f"Load error: {p}")
                return False
        return True

    result = [
        p
        for p in tk.utils.tqdm(list(it), desc="listup", disable=not use_tqdm)
        if _is_valid_file(p)
    ]
    return result, errors


def extract1000(X, y, num_classes):
    """クラスごとに均等に合計1000件を取得する。

    References:
        <https://github.com/mastnk/train1000>

    """
    num_data = 1000
    num_per_class = num_data // num_classes

    index_list = []
    for c in range(num_classes):
        index_list.extend(np.where(y == c)[0][:num_per_class])
    assert len(index_list) == num_data

    return X[index_list], y[index_list]


def cv_indices(X, y, cv_count, cv_index, split_seed, stratify=None):
    """Cross validationのインデックスを返す。

    Args:
        X: 入力データ。
        y: 出力データ。
        cv_count (int): 分割数。
        cv_index (int): 何番目か。
        split_seed (int): 乱数のseed。
        stratify (bool or None): StratifiedKFoldにするならTrue。

    Returns:
        tuple: train_indices, val_indices

    """
    if stratify is None:
        stratify = isinstance(y, np.ndarray) and len(y.shape) == 1
    cv = (
        sklearn.model_selection.StratifiedKFold
        if stratify
        else sklearn.model_selection.KFold
    )
    cv = cv(cv_count, shuffle=True, random_state=split_seed)
    train_indices, val_indices = list(cv.split(X, y))[cv_index]
    return train_indices, val_indices


def to_categorical(num_classes):
    """クラスラベルのone-hot encoding化を行う関数を返す。"""

    def _to_categorical(y):
        assert len(y.shape) == 1
        cat = np.zeros((len(y), num_classes), dtype=np.float32)
        cat[np.arange(len(y)), y] = 1
        return cat

    return _to_categorical


def print_scores(
    precisions, recalls, fscores, supports, class_names=None, print_fn=None
):
    """適合率・再現率などをprintする。(classification_report風。)"""
    assert len(precisions) == len(recalls)
    assert len(precisions) == len(fscores)
    assert len(precisions) == len(supports)
    if class_names is None:
        class_names = [f"class{i:02d}" for i in range(len(precisions))]
    print_fn = print_fn or tk.log.get(__name__).info

    print_fn("                   適合率  再現率  F値    件数")
    # .......'0123456789abcdef:  0.123   0.123   0.123  0123456'

    for cn, prec, rec, f1, sup in zip(
        class_names, precisions, recalls, fscores, supports
    ):
        print_fn(f"{cn:16s}:  {prec:.3f}   {rec:.3f}   {f1:.3f}  {sup:7d}")

    cn = "avg / total"
    prec = np.average(precisions, weights=supports)
    rec = np.average(recalls, weights=supports)
    f1 = np.average(fscores, weights=supports)  # sklearnで言うaverage='weighted'
    sup = np.sum(supports)
    print_fn(f"{cn:16s}:  {prec:.3f}   {rec:.3f}   {f1:.3f}  {sup:7d}")


def top_k_accuracy(y_true, proba_pred, k=5):
    """Top-K accuracy。"""
    assert len(y_true.shape) == 1
    assert len(proba_pred.shape) == 2
    best_k = np.argsort(proba_pred, axis=1)[:, -k:]
    return np.mean([y in best_k[i, :] for i, y in enumerate(y_true)])


def print_classification_metrics(y_true, proba_pred, average="macro", print_fn=None):
    """分類の指標色々を表示する。"""
    try:
        print_fn = print_fn or tk.log.get(__name__).info
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
            print_fn(f"Accuracy:  {acc:.3f} (Error: {1 - acc:.3f})")
            print_fn(f"F1-score:  {f1:.3f}")
            print_fn(f"AUC:       {auc:.3f}")
            print_fn(f"AP:        {ap:.3f}")
            print_fn(f"Precision: {prec:.3f}")
            print_fn(f"Recall:    {rec:.3f}")
            print_fn(f"Logloss:   {logloss:.3f}")
        else:  # multiclass
            assert true_type == "multiclass"
            assert pred_type == "continuous-multioutput"
            num_classes = np.max(y_true) + 1
            labels = list(range(num_classes))
            ohe_true = to_categorical(num_classes)(np.asarray(y_true))
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
            print_fn(f"Accuracy:   {acc:.3f} (Error: {1 - acc:.3f})")
            print_fn(f"F1-{average:5s}:   {f1:.3f}")
            print_fn(f"AUC-{average:5s}:  {auc:.3f}")
            print_fn(f"AP-{average:5s}:   {ap:.3f}")
            print_fn(f"Prec-{average:5s}: {prec:.3f}")
            print_fn(f"Rec-{average:5s}:  {rec:.3f}")
            print_fn(f"Logloss:    {logloss:.3f}")
    except BaseException:
        tk.log.get(__name__).warning(
            "Error: print_classification_metrics", exc_info=True
        )


def print_regression_metrics(y_true, y_pred, print_fn=None):
    """回帰の指標色々を表示する。"""
    try:
        print_fn = print_fn or tk.log.get(__name__).info
        y_mean = np.tile(np.mean(y_pred), len(y_true))
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
        rmseb = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_mean))
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        maeb = sklearn.metrics.mean_absolute_error(y_true, y_mean)
        print_fn(f"R^2:      {r2:.3f}")
        print_fn(f"RMSE:     {rmse:.3f} (base: {rmseb:.3f})")
        print_fn(f"MAE:      {mae:.3f} (base: {maeb:.3f})")
        # RMSE/MAEが1.253より小さいか大きいかで分布の予想がちょっと出来る
        # https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/#how-to-check-rmse-mae-summary
        print_fn(f"RMSE/MAE: {rmse / mae:.3f}")
    except BaseException:
        tk.log.get(__name__).warning("Error: print_regression_metrics", exc_info=True)


@tk_log.trace()
def search_threshold(y_true, y_pred, thresholds, score_fn, direction, cv=10):
    """閾値探索。

    Args:
        y_true (np.ndarray): 答え。
        y_pred (np.ndarray): 予測結果。
        thresholds (np.ndarray): 閾値の配列。
        score_fn (callable): 答えと予測結果と閾値を受け取り、スコアを返す関数。
        direction (str): 'minimize' or 'maximize'
        cv (int): cross validationの分割数。

    Returns:
        tuple: スコアと閾値

    """
    assert direction in ("minimize", "maximize")

    @joblib.delayed
    def _search(fold):
        tr_indices, val_indices = cv_indices(
            y_true, y_true, cv, fold, split_seed=123, stratify=False
        )
        tr_t, tr_p = y_true[tr_indices], y_pred[tr_indices]
        val_t, val_p = y_true[val_indices], y_pred[val_indices]

        best_score = np.inf if direction == "minimize" else -np.inf
        best_th = None
        for th in thresholds:
            s = score_fn(tr_t, tr_p, th)
            if (direction == "minimize" and s < best_score) or (
                direction == "maximize" and s > best_score
            ):
                best_score = s
                best_th = th

        val_score = score_fn(val_t, val_p, best_th)
        tk.log.get(__name__).info(
            f"fold#{fold}: score={best_score:.4f} val_score={val_score:.4f} threshold={best_th:.4f}"
        )
        return val_score, best_th

    with joblib.Parallel(n_jobs=-1, backend="threading") as parallel:
        scores, ths = zip(*parallel([_search(fold) for fold in range(cv)]))

    score = np.mean(scores)
    th = np.mean(ths)
    tk.log.get(__name__).info(f"mean: score={score:.4f} threshold={th:.4f}")
    return score, th
