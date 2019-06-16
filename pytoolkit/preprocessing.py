"""前処理関連。"""

import numpy as np
import sklearn.base
import sklearn.preprocessing


class ClipTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """指定した値でnp.clipするだけのTransformer。"""

    def __init__(self, lower_bound=-2, upper_bound=+2):
        assert lower_bound < upper_bound
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X, y=None):
        del y
        return np.clip(X, self.lower_bound, self.upper_bound)


class PercentileClipTransformer(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    """指定percentileの値でnp.clipするだけのTransformer。"""

    def __init__(self, lower_percentile=5, upper_percentile=95):
        assert 0 <= lower_percentile < upper_percentile <= 100
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bound_ = -np.inf
        self.upper_bound_ = np.inf

    def fit(self, X, y=None):
        del y
        self.lower_bound_, self.upper_bound_ = np.nanpercentile(
            X, [self.lower_percentile, self.upper_percentile], axis=0, keepdims=True
        )
        return self

    def transform(self, X, y=None):
        import pandas as pd

        del y
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            df = X
            X = X.values
        else:
            df = None

        for c in range(len(self.lower_bound_)):
            X[np.isnan(X[:, c]), c] = 0
        X = np.clip(X, self.lower_bound_, self.upper_bound_)
        X[np.isnan(X)] = np.nan

        if df is not None:
            for c in range(len(self.lower_bound_)):
                df.iloc[:, c] = X[:, c]
            return df
        else:
            return X


class HistgramCountTransformer(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    """bins個の値に量子化してCountEncodingして(NNなど用に)log1p。"""

    def __init__(self, bins=10, lower_percentile=5, upper_percentile=95):
        assert 0 <= lower_percentile < upper_percentile <= 100
        self.bins = bins
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.hist_ = None
        self.bin_edges_ = None

    def fit(self, X, y=None):
        import pandas as pd

        del y
        if isinstance(X, pd.DataFrame):
            X = X.values

        # 列ごとにnp.histogram
        self.hist_ = np.empty((X.shape[1], self.bins))
        self.bin_edges_ = np.empty((X.shape[1], self.bins + 1))
        for col in range(X.shape[1]):
            # percentileによるclipping
            q = [self.lower_percentile, self.upper_percentile]
            lb, ub = np.nanpercentile(X[:, col], q, axis=0, keepdims=True)
            # np.histgram
            self.hist_[col], self.bin_edges_[col] = np.histogram(
                np.clip(X[~np.isnan(X[:, col]), col], lb, ub), bins=self.bins
            )
        self.hist_ = np.log1p(self.hist_)
        # 範囲外対策
        self.bin_edges_[:, 0] = -np.inf
        self.bin_edges_[:, -1] = np.inf

        return self

    def transform(self, X, y=None):
        import pandas as pd

        del y
        X = X.copy()
        X_dst = X.iloc if isinstance(X, pd.DataFrame) else X
        columns = len(X.columns) if isinstance(X, pd.DataFrame) else X.shape[1]
        for col in range(columns):
            indices = np.digitize(X_dst[:, col], self.bin_edges_[col]) - 1
            indices = np.clip(indices, 0, self.hist_.shape[1] - 1)
            values = self.hist_[col, indices].copy()
            values[np.isnan(X_dst[:, col])] = np.nan  # digitizeがnanもindexにしてしまうのでnanに戻す
            X_dst[:, col] = values

        return X
