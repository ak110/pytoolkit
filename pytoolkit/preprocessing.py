"""前処理関連。"""

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.preprocessing


class ClipTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """np.clipするだけのTransformer。"""

    def __init__(self, lower_percentile=1, upper_percentile=99):
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
        del y
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.iloc[:, :] = np.clip(X.values, self.lower_bound_, self.upper_bound_)
            return X
        else:
            return np.clip(X, self.lower_bound_, self.upper_bound_)

