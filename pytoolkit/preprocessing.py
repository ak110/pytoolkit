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

    def __init__(self, lower_percentile=2.5, upper_percentile=97.5):
        assert 0 <= lower_percentile < upper_percentile <= 100
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bound_ = -np.inf
        self.upper_bound_ = np.inf

    def fit(self, X, y=None):
        del y
        self.lower_bound_, self.upper_bound_ = np.percentile(
            X, [self.lower_percentile, self.upper_percentile], axis=0, keepdims=True
        )
        return self

    def transform(self, X, y=None):
        import pandas as pd

        del y
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.iloc[:, :] = np.clip(X.values, self.lower_bound_, self.upper_bound_)
            return X
        else:
            return np.clip(X, self.lower_bound_, self.upper_bound_)
