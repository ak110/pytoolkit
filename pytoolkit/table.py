"""pandasなどなど関連。"""
import warnings

import numpy as np
import sklearn.utils

import pytoolkit as tk


def safe_apply(s, fn):
    """nan以外にのみapply"""
    import pandas as pd

    return s.apply(lambda x: x if pd.isnull(x) else fn(x))


def analyze(df):
    """中身を適当に分析してDataFrameに詰めて返す。"""
    import pandas as pd

    if isinstance(df, pd.DataFrame):
        numeric_columns = df.select_dtypes(include=np.number).columns
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
        # はずれ値のはずれ度合いを見るためにRobustScalerした結果の絶対値を見てみる。
        df_result["outlier_size"] = np.nan
        df_result.loc[numeric_columns, "outlier_size"] = (
            tk.preprocessing.SafeRobustScaler(clip_range=None)
            .fit_transform(df.loc[:, numeric_columns])
            .fillna(0)
            .abs()
            .max()
            .round(decimals=1)
        )
        return df_result
    else:
        raise NotImplementedError()


def compare(df1, df2):
    """同じ列を持つ二つのdfの値を色々比べた結果をdfに入れて返す。"""
    assert (df1.columns == df2.columns).all()
    import pandas as pd

    std = (df1.std() + df2.std()) / 2
    df_result = pd.DataFrame(index=df1.columns)
    df_result["mean_ae/std"] = np.abs(df1.mean() - df2.mean()) / std
    df_result["median_ae/std"] = np.abs(df1.median() - df2.median()) / std
    df_result["mode1"] = df1.mode().transpose()[0]
    df_result["mode2"] = df2.mode().transpose()[0]

    df_result = df_result.sort_values("median_ae/std", ascending=False)
    return df_result


def shuffled_score(score_fn, X, c, y, n_iter=5, random_state=None):
    """Permutation Importanceのための処理。

    c列をシャッフルしたときのスコアの平均値を返す。

    Args:
        - score_fn (callable): X, yを受け取りスコアを返す関数。
        - X (pd.DataFrame): 入力データ
        - c (str): 対象の列
        - y (np.ndarray): ラベル
        - n_iter (int): 繰り返し回数
        - random_state: seed

    """
    import pandas as pd

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


def latlon_distance(lat1, lon1, lat2, lon2):
    """2地点間の距離。

    Args:
        lat1: 地点1の緯度[°]
        lon1: 地点1の経度[°]
        lat2: 地点2の緯度[°]
        lon2: 地点2の経度[°]

    Returns:
        float: 距離[km]

    References:
        <https://keisan.casio.jp/exec/system/1257670779>

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
