"""機械学習関連。"""
import pathlib
import typing

import joblib
import numpy as np
import sklearn.base
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf

import pytoolkit as tk


def listup_classification(
    dirpath,
    class_names: typing.Sequence[str] = None,
    use_tqdm: bool = True,
    check_image: bool = False,
) -> typing.Tuple[typing.List[str], np.ndarray, np.ndarray]:
    """画像分類でよくある、クラス名ディレクトリの列挙。クラス名の配列, X, yを返す。

    Args:
        class_names: クラス名の配列
        use_tqdm: tqdmを使用するか否か
        check_image: 画像として読み込みチェックを行い、読み込み可能なファイルのみ返すか否か (遅いので注意)

    Returns:
        class_names, X, y

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

        class_names = sorted(p.name for p in dirpath.iterdir() if _is_valid_classdir(p))

    class_names = list(class_names)

    # 各クラスのデータを列挙
    X: list = []
    y: list = []
    errors: list = []
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
            except Exception:
                errors.append(f"Load error: {p}")
                return False
        return True

    result = [
        p
        for p in tk.utils.tqdm(list(it), desc="listup", disable=not use_tqdm)
        if _is_valid_file(p)
    ]
    return result, errors


def cv_indices(X, y, cv_count, cv_index, split_seed, stratify=None):
    """Cross validationのインデックスを返す。"""
    folds = get_folds(X, y, cv_count, split_seed, stratify)
    train_indices, val_indices = folds[cv_index]
    return train_indices, val_indices


def get_folds(X, y, cv_count: int, split_seed: int, stratify: bool = None):
    """Cross validationのインデックスを返す。

    Args:
        X: 入力データ。
        y: 出力データ。
        cv_count: 分割数。
        split_seed: 乱数のseed。
        stratify: StratifiedKFoldにするならTrue。

    Returns:
        list of tuple(train_indices, val_indices): インデックス

    """
    if stratify is None:
        stratify = isinstance(y, np.ndarray) and len(y.shape) == 1
    cv = (
        sklearn.model_selection.StratifiedKFold
        if stratify
        else sklearn.model_selection.KFold
    )
    cv = cv(cv_count, shuffle=True, random_state=split_seed)
    folds = list(cv.split(X, y))
    return folds


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


def search_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray,
    score_fn: typing.Callable[[np.ndarray, np.ndarray, float], float],
    direction: str,
    cv: int = 10,
) -> typing.Tuple[float, float]:
    """閾値探索。

    Args:
        y_true: 答え。
        y_pred: 予測結果。
        thresholds: 閾値の配列。
        score_fn: 答えと予測結果と閾値を受け取り、スコアを返す関数。
        direction: 'minimize' or 'maximize'
        cv: cross validationの分割数。

    Returns:
        スコア, 閾値

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
        best_th: typing.Optional[float] = None
        for th in thresholds:
            s = score_fn(tr_t, tr_p, th)
            if (direction == "minimize" and s < best_score) or (
                direction == "maximize" and s > best_score
            ):
                best_score = s
                best_th = th
        assert best_th is not None

        val_score = score_fn(val_t, val_p, best_th)
        tk.log.get(__name__).info(
            f"fold#{fold}: score={best_score:.4f} val_score={val_score:.4f} threshold={best_th:.4f}"
        )
        return val_score, best_th

    with tk.log.trace("search_threshold"):
        with joblib.Parallel(n_jobs=-1, backend="threading") as parallel:
            scores, ths = zip(*parallel([_search(fold) for fold in range(cv)]))

        score = np.mean(scores)
        th = np.mean(ths)
        tk.log.get(__name__).info(f"mean: score={score:.4f} threshold={th:.4f}")
        return score, th


def top_k_accuracy(y_true, y_pred, k=5):
    """Top-K accuracy。"""
    assert len(y_true) == len(y_pred)
    assert y_true.ndim in (1, 2)
    assert y_pred.ndim == 2
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=-1)
    best_k = np.argsort(y_pred, axis=1)[:, -k:]
    return np.mean([y in best_k[i, :] for i, y in enumerate(y_true)])


def mape(y_true, y_pred):
    """MAPE(mean absolute percentage error)。"""
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    assert len(y_true) == len(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calibrate_proba(proba: typing.Union[np.ndarray, tf.Tensor], beta: np.ndarray):
    """不均衡データなどの分類の確信度の補正。

    Args:
        proba: 補正する確信度。shape=(samples, classes)
        beta: 補正する係数。shape=(classes,)

    Returns:
        補正結果。shape=(samples, classes)

    References:
        - <https://quinonero.net/Publications/predicting-clicks-facebook.pdf>

    """
    # return proba / (proba + (1 - proba) / beta)
    # return (beta * proba) / (beta * proba - proba + 1)
    bp = proba * beta / np.sum(beta)
    return bp / (bp + 1 - proba)


def get_effective_class_weights(
    samples_per_classes: typing.Sequence[int], beta: float = 0.9999
):
    """Class-Balanced Loss風のclass_weightsを作成して返す。<https://arxiv.org/abs/1901.05555>

    Args:
        samples_per_classes: クラスごとのサンプルサイズ (shape=(num_classes,))
        beta: ハイパーパラメーター

    """
    effectives = 1.0 - beta ** np.asarray(samples_per_classes)
    weights = (1 - beta) / effectives
    weights = weights * (len(samples_per_classes) / weights.sum())
    return weights
