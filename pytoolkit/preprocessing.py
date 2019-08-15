"""前処理関連。"""

import numpy as np
import scipy.special
import scipy.stats
import sklearn.base
import sklearn.pipeline
import sklearn.preprocessing


class NullTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """何もしないTransformer。"""

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X, y=None):
        del y
        return X


class DataFrameToNDArray(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """DataFrameをndarrayに変換するだけのTransformer。"""

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X, y=None):
        del y
        return X.values


class ResidualTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """TransformedTargetRegressorなどと組み合わせて残差を学習するようにするためのTransformer。"""

    def __init__(self, pred_train, pred_test):
        assert len(pred_train) != len(pred_test)  # train/testが同一件数の場合は未実装
        self.pred_train = pred_train
        self.pred_test = pred_test

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X, y=None):
        del y
        return X - self._get_base(X)

    def inverse_transform(self, X, y=None):
        del y
        return X + self._get_base(X)

    def _get_base(self, X):
        if len(X) == len(self.pred_train):
            return self.pred_train
        elif len(X) == len(self.pred_test):
            return self.pred_test
        else:
            raise ValueError()


class Normalizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """列ごとに値の傾向を見てできるだけいい感じにスケーリングなどをする。"""

    def __init__(self, cols=None, return_df=True, clip_range=(-7, +7)):
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.clip_range = clip_range
        self.params_ = None

    def fit(self, X, y=None):
        del y
        self.params_ = {}
        for c in _get_cols(self.cols, X):
            values = _get_col_values(X, c).astype(np.float32)
            assert not np.isinf(values).any(), f"Invalid value: c={c} values={values}"

            # nanの削除
            values = values[~np.isnan(values)]
            # 外れ値のクリッピング
            mean, std3 = np.mean(values), np.std(values) * 3
            lower, upper = np.percentile(values, (1, 99))
            lower = min(lower, mean - std3)
            upper = max(lower, mean + std3)
            if lower != upper:
                values = np.clip(values, lower, upper)

            nunique = len(np.unique(values))
            if nunique <= len(values) * 0.01 or np.std(values) <= 0:
                # 値が離散的なら適当にスケーリング
                cr = self.clip_range or [-7, +7]
                a2 = min(np.sqrt(nunique), cr[1] - cr[0] - 1)
                self.params_[c] = {
                    "lmbda": None,
                    "center": (values.max() + values.min()) / 2,
                    "scale": (values.max() - values.min()) / a2,
                }
            else:
                # 値が連続的ならYeo-Johnson power transformation ＆ standalization
                skew = scipy.stats.skew(values)
                if skew <= 0.75:
                    lmbda = None
                else:
                    values_y, lmbda = scipy.stats.yeojohnson(values)
                    if lmbda >= 0:
                        values = values_y
                    else:
                        lmbda = None  # lmbda < 0は何かあやしいのでとりあえずスキップ。。
                self.params_[c] = {
                    "lmbda": lmbda,
                    "center": np.median(values),
                    "scale": np.std(values),
                }

        return self

    def transform(self, X, y=None):
        del y
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
            for c, p in self.params_.items():
                X[:, c] = self._transform_column(X, c, **p)
        else:
            for c, p in self.params_.items():
                X[c] = self._transform_column(X, c, **p)
        return X if self.return_df or isinstance(X, np.ndarray) else X.values

    def _transform_column(self, X, c, lmbda, center, scale):
        s = _get_col_values(X, c).astype(np.float32)
        if lmbda is not None:
            s = scipy.stats.yeojohnson(s, lmbda)
        s -= center
        s /= 1 if scale == 0 else scale
        if self.clip_range is not None:
            s = np.clip(s, *self.clip_range)
        return s


class CountEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Count Encoding。

    Category Encoders <http://contrib.scikit-learn.org/categorical-encoding/> 風のインターフェースにしてみる。
    (全部は作っていない)

    """

    def __init__(self, cols=None, return_df=True, handle_unknown="return_nan"):
        assert handle_unknown in ("error", "value", "return_nan")
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.maps_ = None

    def fit(self, X, y=None):
        del y
        self.maps_ = {}
        for c in self.cols or X.columns.values:
            self.maps_[c] = X[c].value_counts(dropna=False).astype(np.float32).to_dict()
            if np.nan in self.maps_[c]:
                if self.handle_unknown == "error":
                    raise ValueError(f"Column '{c}' has NaN!")
                if self.handle_unknown == "return_nan":
                    self.maps_[c][np.nan] = np.nan
        return self

    def transform(self, X, y=None):
        del y
        X = X.copy()
        cols = self.cols or X.columns.values
        for c in cols:
            X[c] = X[c].map(self.maps_[c]).astype(np.float32)
        return X if self.return_df else X.values


class TargetEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Target Encoding。"""

    def __init__(self, cols=None, return_df=True, *, folds):
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.folds = folds
        self.train_df_ = None
        self.train_maps_ = None
        self.test_maps_ = None

    def fit(self, X, y):
        assert y is not None
        self.train_df_ = X
        cols = self.cols or X.columns.values.tolist()
        # train用のencodingを作る
        assert "__target__" not in cols  # 手抜き
        Xy = X.copy()
        Xy["__target__"] = y
        self.train_maps_ = {c: np.repeat(None, len(self.folds)) for c in cols}
        for fold, (train_indices, _) in enumerate(self.folds):
            Xy_t = Xy.iloc[train_indices]
            for c in cols:
                d = Xy_t.groupby(c)["__target__"].mean().to_dict()
                self.train_maps_[c][fold] = d
        # test用のencodingを作る
        self.test_maps_ = {c: Xy.groupby(c)["__target__"].mean() for c in cols}
        return self

    def transform(self, X, y=None):
        del y
        X = X.copy()
        cols = self.cols or X.columns.values
        if X.equals(self.train_df_):
            for c in cols:
                for fold, (_, val_indices) in enumerate(self.folds):
                    X.loc[val_indices, c] = X.loc[val_indices, c].map(
                        self.train_maps_[c][fold]
                    )
                X[c] = X[c].astype(np.float32)
        else:
            for c in cols:
                X[c] = X[c].map(self.test_maps_[c]).astype(np.float32)
        return X if self.return_df else X.values


class TargetOrderEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """foldを切らないTarget Encoding。"""

    def __init__(self, cols=None, return_df=True, min_samples_leaf=1):
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.min_samples_leaf = min_samples_leaf
        self.maps_ = None

    def fit(self, X, y):
        assert y is not None

        cols = self.cols or X.columns.values.tolist()
        assert "__target__" not in cols  # 手抜き
        Xy = X.copy()
        Xy["__target__"] = y

        self.maps_ = {c: self._target(Xy, c) for c in cols}
        return self

    def _target(self, Xy, c):
        g = Xy.groupby(c)["__target__"]
        s = g.mean()
        if self.min_samples_leaf > 1:
            s = s[g.count() >= self.min_samples_leaf]
        s = s.argsort() + 1
        return s.to_dict()

    def transform(self, X, y=None):
        del y
        X = X.copy()
        cols = self.cols or X.columns.values
        for c in cols:
            X[c] = X[c].map(self.maps_[c]).astype(np.float32)
        return X if self.return_df else X.values


def encode_binary(s, true_value, false_value):
    """列の2値化。"""
    s2 = s.map({true_value: True, false_value: False})
    assert s2.notnull().all(), f"Convert error: {s[s2.isnull()].value_counts()}"
    return s2.astype(bool)


def encode_target(s, s_train, y_train, min_samples_leaf=20, smoothing=10.0):
    """Target Encoding。"""
    assert min_samples_leaf >= 2
    import pandas as pd

    data = np.repeat(np.nan, len(s))
    unique_values = s_train.unique()
    y_train = y_train.astype(np.float32)
    prior = y_train.mean()
    for v in unique_values:
        m = s.isnull() if pd.isnull(v) else (s == v)
        tm = s_train.isnull() if pd.isnull(v) else (s_train == v)
        tm_size = tm.sum()
        if tm_size >= min_samples_leaf:
            target = y_train[tm].mean()
            data[m] = smooth(target, prior, tm_size, min_samples_leaf, smoothing)

    data[np.isnan(data)] = prior

    return pd.Series(data=data, index=s.index, dtype=np.float32)


def smooth(target, prior, target_samples, min_samples_leaf=1, smoothing=1.0):
    """Target Encodingのsmoothing。

    Args:
        target (np.ndarray): Target Encodingの値(カテゴリごとの平均値)
        prior (np.ndarray): target全体の平均
        target_samples (int): カテゴリごとの要素の数
        min_samples_leaf (int): カテゴリごとの要素の数の最小値
        smoothing (float): smoothingの強さ

    """
    smoove = 1 / (1 + np.exp(-(target_samples - min_samples_leaf) / smoothing))
    return target * smoove + prior * (1 - smoove)


def _get_cols(cols, X):
    """DataFrameとndarrayを統一的に扱うためのヘルパー: 列の列挙"""
    if cols is not None:
        return cols
    if isinstance(X, np.ndarray):
        return list(range(X.shape[1]))
    return X.columns.values


def _get_col_values(X, c):
    """DataFrameとndarrayを統一的に扱うためのヘルパー: ndarrayの取得"""
    if isinstance(X, np.ndarray):
        return X[:, c]
    return X[c].values
