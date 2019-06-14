"""前処理関連。"""

import numpy as np
import sklearn.base
import sklearn.preprocessing


class ClipTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """np.clipするだけのTransformer。"""

    def __init__(self, min_value=-3, max_value=+3):
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X, y=None):
        del y
        return np.clip(X, self.min_value, self.max_value)
