"""pandasなどなど関連。"""
from __future__ import annotations

import html
import typing
import warnings

import numpy as np
import pandas as pd
import sklearn.utils

import pytoolkit as tk


def target_encoding(
    values: typing.Union[pd.Series, np.ndarray],
    values_train: typing.Union[pd.Series, np.ndarray],
    target_train: np.ndarray,
    min_samples_leaf: int = 3,
    smoothing: float = 1.0,
):
    """ターゲットエンコーディング。"""
    d = make_target_encoding_map(
        values_train, target_train, min_samples_leaf, smoothing
    )
    return pd.Series(values).map(d)


def make_target_encoding_map(
    values_train: typing.Union[pd.Series, np.ndarray],
    target_train: np.ndarray,
    min_samples_leaf: int = 3,
    smoothing: float = 1.0,
) -> typing.Dict[typing.Any, np.float32]:
    """ターゲットエンコーディングの変換用dictの作成。"""
    df_tmp = pd.DataFrame()
    df_tmp["values"] = values_train
    df_tmp["target"] = target_train
    g = df_tmp.groupby("values")["target"]
    s = g.mean()
    c = g.count()
    prior = df_tmp["target"].mean()
    smoove = 1 / (1 + np.exp(-(c - min_samples_leaf) / smoothing))
    smoothed = prior * (1 - smoove) + s.values * smoove
    smoothed[c <= min_samples_leaf] = prior
    d = dict(zip(s.index.values, np.float32(smoothed)))
    return d


def safe_apply(s: pd.Series, fn) -> pd.Series:
    """nan以外にのみapply"""
    return s.apply(lambda x: x if pd.isnull(x) else fn(x))


def add_col(
    df: pd.DataFrame, column_name: str, values: typing.Sequence[typing.Any]
) -> None:
    """上書きしないようにチェックしつつ列追加。"""
    if column_name in df:
        raise ValueError(f"Column '{column_name}' already exists.")
    df[column_name] = values


def add_cols(
    df: pd.DataFrame,
    column_names: typing.List[str],
    values: typing.Sequence[typing.Any],
) -> None:
    """上書きしないようにチェックしつつ列追加。"""
    for column_name in column_names:
        if column_name in df:
            raise ValueError(f"Column '{column_name}' already exists.")
    df[column_names] = values


def group_columns(
    df: pd.DataFrame, cols: typing.Sequence[str] = None
) -> typing.Dict[str, typing.List[str]]:
    """列を型ごとにグルーピングして返す。

    Args:
        df: DataFrame
        cols: 対象の列名の配列

    Returns:
        種類ごとの列名のlist
            - "binary": 二値列
            - "numeric": 数値列
            - "categorical": カテゴリ列(など)
            - "unknown": その他

    """
    binary_cols = []
    numeric_cols = []
    categorical_cols = []
    unknown_cols = []
    for c in cols or df.columns.values:
        if pd.api.types.is_bool_dtype(df[c].dtype):
            binary_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c].dtype):
            numeric_cols.append(c)
        elif pd.api.types.is_categorical_dtype(
            df[c].dtype
        ) or pd.api.types.is_object_dtype(df[c].dtype):
            categorical_cols.append(c)
        else:
            unknown_cols.append(c)
    return {
        "binary": binary_cols,
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "unknown": unknown_cols,
    }


def eda(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """色々調べて表示する。(jupyter用)"""
    from IPython.display import display, HTML

    display(HTML(eda_html(df_train, df_test)))


def eda_html(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """色々調べて結果をHTML化して返す。"""
    result = ""

    train_cols = group_columns(df_train)
    test_cols = group_columns(df_test)

    def td_count(count, all_count):
        if count < all_count * 0.01:
            text = f"<span style='text-decoration: underline;'>{count:d}</span>"
        else:
            text = f"{count / all_count:.0%}"
        return f"<td style='text-align: right;'>{text}</td>"

    def td_value(value):
        return f"<td style='text-align: right;'>{value:,.1f}</td>"

    binary_cols = np.unique(train_cols["binary"] + test_cols["binary"])
    if len(binary_cols) > 0:
        result += (
            "<h2>binary</h2>"
            "<table>"
            "<thead>"
            "<tr>"
            "<td></td>"
            "<td colspan=1 style='text-align: left;'>train</td>"
            "<td colspan=1 style='text-align: left;'>test</td>"
            "</tr>"
            "<tr>"
            "<td style='text-align: left;'>列名</td>"
            "<td style='text-align: center;'>True%</td>"
            "<td style='text-align: center;'>True%</td>"
            "</tr>"
            "</thead>"
            "<tbody>"
        )
        for c in binary_cols:
            result += "<tr>"
            result += f"<td style='text-align: left;'>{html.escape(c)}</td>"
            for df, exists in [
                (df_train, c in train_cols["binary"]),
                (df_test, c in test_cols["binary"]),
            ]:
                if exists:
                    result += td_count(df[c].sum(), len(df))
                else:
                    result += "<td style='text-align: right;'></td>"
            result += "</tr>"
        result += "</tbody>"
        result += "</table>"

    numeric_cols = np.unique(train_cols["numeric"] + test_cols["numeric"])
    if len(numeric_cols) > 0:
        result += (
            "<h2>numeric</h2>"
            "<table>"
            "<thead>"
            "<tr>"
            "<td></td>"
            "<td colspan=4 style='text-align: left;'>train</td>"
            "<td colspan=4 style='text-align: left;'>test</td>"
            "</tr>"
            "<tr>"
            "<td style='text-align: left;'>列名</td>"
            "<td style='text-align: center;'>null</td>"
            "<td style='text-align: center;'>nunique</td>"
            "<td style='text-align: center;'>mean</td>"
            "<td style='text-align: center;'>std</td>"
            "<td style='text-align: center;'>null</td>"
            "<td style='text-align: center;'>nunique</td>"
            "<td style='text-align: center;'>mean</td>"
            "<td style='text-align: center;'>std</td>"
            "</tr>"
            "</thead>"
            "<tbody>"
        )
        for c in numeric_cols:
            result += "<tr>"
            result += f"<td style='text-align: left;'>{html.escape(c)}</td>"
            for df, exists in [
                (df_train, c in train_cols["numeric"]),
                (df_test, c in test_cols["numeric"]),
            ]:
                if exists:
                    result += td_count(df[c].isnull().sum(), len(df))
                    result += td_count(df[c].nunique(), len(df))
                    result += td_value(df[c].mean())
                    result += td_value(df[c].std())
                else:
                    result += "<td></td><td></td><td></td><td></td>"
            result += "</tr>"
        result += "</tbody>"
        result += "</table>"

    categorical_cols = np.unique(train_cols["categorical"] + test_cols["categorical"])
    if len(categorical_cols) > 0:
        result += (
            "<h2>categorical</h2>"
            "<table>"
            "<thead>"
            "<tr>"
            "<td style='text-align: left;'>列名</td>"
            "<td style='text-align: left;'>値</td>"
            "</tr>"
            "</thead>"
            "<tbody>"
        )
        for c in categorical_cols:
            result += (
                "<tr>"
                f"<td style='text-align: left; vertical-align: top;'>{html.escape(c)}</td>"
                "<td>"
                "<table>"
                "<tbody>"
                "<tr>"
            )
            if c in train_cols["categorical"]:
                values1 = df_train[c].value_counts().to_dict()
                nulls = df_train[c].isnull().sum()
                if nulls > 0:
                    values1[""] = nulls
            else:
                values1 = {}
            if c in test_cols["categorical"]:
                values2 = df_test[c].value_counts().to_dict()
                nulls = df_test[c].isnull().sum()
                if nulls > 0:
                    values2[""] = nulls
            else:
                values2 = {}
            for v in set(list(values1) + list(values2)):
                result += "<tr>"
                result += f"<td style='text-align: left;'>{html.escape(str(v))}</td>"
                result += (
                    td_count(values1[v], len(df_train)) if v in values1 else "<td></td>"
                )
                result += (
                    td_count(values2[v], len(df_test)) if v in values2 else "<td></td>"
                )
                result += "</tr>"
            result += "</tr></tbody></table></td></tr>"
        result += "</tbody></table>"

    unknown_cols = np.unique(train_cols["unknown"] + test_cols["unknown"])
    if len(unknown_cols) > 0:
        result += (
            "<h2>unknown</h2>"
            "<table>"
            "<thead>"
            "<tr>"
            "<td style='text-align: left;'>列名</td>"
            "<td style='text-align: left;'>値</td>"
            "</tr>"
            "</thead>"
            "<tbody>"
        )
        for c in unknown_cols:
            result += (
                "<tr>"
                f"<td style='text-align: left; vertical-align: top;'>{html.escape(c)}</td>"
                "<td>"
                "<table>"
                "<tbody>"
                "<tr>"
            )
            if c in train_cols["unknown"]:
                values1 = df_train[c].value_counts().to_dict()
                nulls = df_train[c].isnull().sum()
                if nulls > 0:
                    values1[""] = nulls
            else:
                values1 = {}
            if c in test_cols["unknown"]:
                values2 = df_test[c].value_counts().to_dict()
                nulls = df_test[c].isnull().sum()
                if nulls > 0:
                    values2[""] = nulls
            else:
                values2 = {}
            for v in set(list(values1) + list(values2)):
                result += "<tr>"
                result += f"<td style='text-align: left;'>{html.escape(str(v))}</td>"
                result += (
                    td_count(values1[v], len(df_train)) if v in values1 else "<td></td>"
                )
                result += (
                    td_count(values2[v], len(df_test)) if v in values2 else "<td></td>"
                )
                result += "</tr>"
            result += "</tr></tbody></table></td></tr>"
        result += "</tbody></table>"

    return result


def analyze(df: pd.DataFrame):
    """中身を適当に分析してDataFrameに詰めて返す。"""
    if isinstance(df, pd.DataFrame):
        df_result = pd.DataFrame(index=df.columns)
        df_result["dtype"] = df.dtypes
        df_result["null"] = df.isnull().sum()
        df_result["nunique"] = df.nunique()
        df_result["min"] = df.min()
        df_result["median"] = df.median()
        df_result["max"] = df.max()
        df_result["mode"] = df.mode().transpose()[0]
        df_result["mean"] = df.mean()
        df_result["std"] = df.std()
        # # はずれ値のはずれ度合いを見るためにRobustScalerした結果の絶対値を見てみる。
        # numeric_columns = df.select_dtypes(include=np.number).columns
        # df_result["outlier_size"] = np.nan
        # df_result.loc[numeric_columns, "outlier_size"] = (
        #     tk.preprocessing.SafeRobustScaler(clip_range=None)
        #     .fit_transform(df.loc[:, numeric_columns])
        #     .fillna(0)
        #     .abs()
        #     .max()
        #     .round(decimals=1)
        # )
        return df_result
    else:
        raise NotImplementedError()


def compare(df1: pd.DataFrame, df2: pd.DataFrame):
    """同じ列を持つ二つのdfの値を色々比べた結果をdfに入れて返す。"""
    assert (df1.columns == df2.columns).all()

    std = (df1.std() + df2.std()) / 2
    df_result = pd.DataFrame(index=df1.columns)
    df_result["mean_ae/std"] = np.abs(df1.mean() - df2.mean()) / std
    df_result["median_ae/std"] = np.abs(df1.median() - df2.median()) / std
    df_result["mode1"] = df1.mode().transpose()[0]
    df_result["mode2"] = df2.mode().transpose()[0]

    df_result = df_result.sort_values("median_ae/std", ascending=False)
    return df_result


def permutation_importance(
    score_fn: typing.Callable[[np.ndarray, np.ndarray], float],
    X: pd.DataFrame,
    y: np.ndarray,
    greater_is_better: bool,
    columns: typing.Sequence[str] = None,
    n_iter: int = 5,
    random_state=None,
    verbose: bool = True,
):
    """Permutation Importanceを算出して返す。

    Args:
        score_fn: X, yを受け取りスコアを返す関数。
        X: 入力データ
        y: ラベル
        greater_is_better: スコアが大きいほど良いならTrue
        columns: 対象の列
        n_iter: 繰り返し回数
        random_state: seed
        verbose: プログレスバーを表示するか否か

    Returns:
        pd.DataFrame: columnとimportanceの列を持つDataFrame

    """
    if columns is None:
        columns = X.columns.values

    base_score = score_fn(X, y)
    tk.log.get(__name__).info(f"Base Score: {base_score:.2f}")

    importances = []
    for c in tk.utils.tqdm(columns, disable=not verbose):
        ss = shuffled_score(
            score_fn=score_fn, X=X, c=c, y=y, n_iter=n_iter, random_state=random_state
        )
        s = base_score - ss if greater_is_better else ss - base_score
        importances.append(s)
        tk.utils.tqdm_write(f"{c:40s}: {s:7.3f}")

    df_importance = pd.DataFrame()
    df_importance["column"] = columns
    df_importance["importance"] = importances
    return df_importance


def shuffled_score(
    score_fn: typing.Callable[[np.ndarray, np.ndarray], float],
    X: pd.DataFrame,
    c: str,
    y: np.ndarray,
    n_iter: int = 5,
    random_state=None,
) -> float:
    """Permutation Importanceのための処理。

    c列をシャッフルしたときのスコアの平均値を返す。

    Args:
        score_fn: X, yを受け取りスコアを返す関数。
        X: 入力データ
        c: 対象の列
        y: ラベル
        n_iter: 繰り返し回数
        random_state: seed

    Returns:
        スコア

    """
    random_state = sklearn.utils.check_random_state(random_state)
    X = X.copy()
    original = X[c]
    scores = []
    nunique = original.nunique(dropna=False)
    if nunique - 1 <= n_iter:
        # 値の種類の数が少ないなら全件チェック
        values = original.value_counts(dropna=False).index.values
        assert len(values) == nunique
        for n in range(1, nunique):
            d = {values[i]: values[(i + n) % nunique] for i in range(nunique)}
            X[c] = original.map(d).astype(original.dtype)
            scores.append(score_fn(X, y))
    else:
        # n_iter回シャッフルしてスコアを算出
        for _ in range(n_iter):
            data = random_state.permutation(original.values)
            X[c] = pd.Series(data=data, dtype=original.dtype)
            scores.append(score_fn(X, y))
    return np.mean(scores)


def latlon_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2地点間の距離。

    Args:
        lat1: 地点1の緯度[°]
        lon1: 地点1の経度[°]
        lat2: 地点2の緯度[°]
        lon2: 地点2の経度[°]

    Returns:
        距離[km]

    References:
        - <https://keisan.casio.jp/exec/system/1257670779>

    """
    r = 6378.137
    d2r = np.pi / 180
    delta_x = lon2 - lon1
    s = np.sin(lat1 * d2r) * np.sin(lat2 * d2r)
    c = np.cos(lat1 * d2r) * np.cos(lat2 * d2r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = r * np.arccos(s + c * np.cos(delta_x * d2r))
    return d
