"""pandasなどなど関連。"""

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
    scores = []
    for _ in range(n_iter):
        data = random_state.permutation(X[c].values)
        X[c] = pd.Series(data=data, dtype=X[c].dtype)
        scores.append(score_fn(X, y))
    return np.mean(scores)
