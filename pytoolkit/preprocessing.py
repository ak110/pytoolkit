"""前処理関連。"""

import numpy as np
import sklearn.base
import sklearn.preprocessing
import sklearn.pipeline
import scipy.special
import scipy.stats


class NullTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """何もしないTransformer。"""

    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X, y=None):
        del y
        return X


class DataFrameTransformerBase(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    """DataFrameに対応したTransformer。派生クラスで_fitと_transformを実装するようにする。"""

    def fit(self, X, y=None):
        import pandas as pd

        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        return self._fit(X, y)

    def transform(self, X, y=None):
        import pandas as pd

        if isinstance(y, pd.DataFrame):
            y = y.values
        elif isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.iloc[:, :] = self._transform(X.values, y)
        elif isinstance(X, pd.Series):
            X = X.copy()
            X.iloc[:] = self._transform(X.values.reshape(-1, 1), y)[:, 0]
        else:
            X = self._transform(X, y)
        return X

    def _fit(self, X, y=None):
        del X, y
        return self

    def _transform(self, X, y=None):
        raise NotImplementedError()


class SafeRobustScaler(DataFrameTransformerBase):
    """nanをnanのままにするRobustScalerのようなもの+np.clip()。"""

    def __init__(self, quantile_range=(25.0, 75.0), clip_range=(-3, +3)):
        assert 0 <= quantile_range[0] < quantile_range[1] <= 100
        self.quantile_range = quantile_range
        self.clip_range = clip_range
        self.center_ = None
        self.scale_ = None

    def _fit(self, X, y=None):
        del y
        X = X.astype(np.float32)
        self.center_ = np.nanmedian(X, axis=0)
        lower, upper = np.nanpercentile(X, self.quantile_range, axis=0)
        scale = np.min([upper - lower, np.nanstd(X, axis=0)], axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def _transform(self, X, y=None):
        assert X.shape[1] == len(self.center_)
        del y
        X = X.copy().astype(np.float32)
        X -= self.center_
        X /= self.scale_
        if self.clip_range is not None:
            X = np.clip(X, self.clip_range[0], self.clip_range[1])
        return X


class HistgramCountEncoder(DataFrameTransformerBase):
    """bins個の値に量子化してCountEncoding+NNなど用適当スケーリング。"""

    def __init__(self, bins=10, percentile_range=(1, 99)):
        assert 0 <= percentile_range[0] < percentile_range[1] <= 100
        self.bins = bins
        self.percentile_range = percentile_range
        self.hist_ = None
        self.bin_edges_ = None

    def _fit(self, X, y=None):
        del y
        # 列ごとにnp.histogram
        self.hist_ = np.empty((X.shape[1], self.bins))
        self.bin_edges_ = np.empty((X.shape[1], self.bins - 1))
        for col in range(X.shape[1]):
            # percentileによるclipping
            lb, ub = np.nanpercentile(
                X[:, col], self.percentile_range, axis=0, keepdims=True
            )
            # np.histgram
            hist, bin_edges = np.histogram(
                np.clip(X[~np.isnan(X[:, col]), col], lb, ub), bins=self.bins
            )
            self.hist_[col] = hist
            self.bin_edges_[col] = bin_edges[1:-1]  # 両端は最小値と最大値なので要らない
        return self

    def _transform(self, X, y=None):
        assert X.shape[1] == len(self.hist_)
        del y
        X = X.copy().astype(np.float32)
        for col in range(X.shape[1]):
            indices = np.digitize(X[:, col], self.bin_edges_[col])
            values = self.hist_[col, indices].copy()
            values[np.isnan(X[:, col])] = np.nan  # digitizeがnanもindexにしてしまうのでnanに戻す
            X[:, col] = values
        return X


class RankGaussEncoder(DataFrameTransformerBase):
    """RankGauss(風)変換 <http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss/>"""

    def __init__(self):
        self.bin_edges_ = None
        self.gauss_ = None

    def _fit(self, X, y=None):
        del y
        self.bin_edges_ = []
        self.gauss_ = []
        for col in range(X.shape[1]):
            # nan以外のユニークな値を取り出してソート
            values = X[:, col].astype(np.float32)
            values = np.sort(np.unique(values[~np.isnan(values)]))
            bin_edges = (values[:-1] + values[1:]) / 2
            self.bin_edges_.append(bin_edges)
            # 正規分布っぽくなるような値を用意
            gauss = scipy.special.erfinv(np.linspace(-1, +1, len(bin_edges) + 3))[1:-1]
            self.gauss_.append(gauss)
        return self

    def _transform(self, X, y=None):
        assert X.shape[1] == len(self.gauss_)
        del y
        X = X.copy().astype(np.float32)
        for col in range(X.shape[1]):
            indices = np.digitize(X[:, col], self.bin_edges_[col])
            values = self.gauss_[col][indices].copy()
            values[np.isnan(X[:, col])] = np.nan  # digitizeがnanもindexにしてしまうのでnanに戻す
            X[:, col] = values
        return X


class Normalizer(DataFrameTransformerBase):
    """列ごとに値の傾向を見てできるだけいい感じにスケーリングなどをする。"""

    def __init__(self):
        self.columns_scaler_ = None
        self.scalers_ = {
            "robust": SafeRobustScaler(),
            "power": sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.PowerTransformer(),
                sklearn.preprocessing.StandardScaler(),
            ),
            "rank": RankGaussEncoder(),
        }

    def _fit(self, X, y=None):
        # 列ごとに傾向を見て使うscalerを決定
        X = X.copy().astype(np.float32)
        self.columns_scaler_ = []
        for col in range(X.shape[1]):
            col_values = X[:, col]

            # 二値っぽければ何もしない
            if np.isin(col_values, (np.float32(0), np.float32(1))).all():
                self.columns_scaler_.append(None)
                continue

            # nanとはずれ値を除去
            col_values = col_values[~np.isnan(col_values)]
            lb, ub = np.percentile(col_values, (1, 99))
            robust_col_values = col_values[
                np.logical_and(lb <= col_values, col_values <= ub)
            ]

            # あまり離散的でなく、かつ外れ値が酷ければRankGauss
            unique_rate = len(np.unique(col_values)) / len(X)
            outlier_size = np.maximum(lb - col_values.min(), col_values.max() - ub) / (
                ub - lb + 1e-7
            )
            if unique_rate >= 0.05 and outlier_size >= 0.3:
                self.columns_scaler_.append("rank")
                continue

            # 正規分布っぽければSafeRobustScaler
            skew = scipy.stats.skew(robust_col_values)
            if skew <= 0.75:
                self.columns_scaler_.append("robust")
                continue

            # 上記以外ならPowerTransformer+SafeRobustScaler
            self.columns_scaler_.append("power")

        self.columns_scaler_ = np.array(self.columns_scaler_)

        # scalerの学習
        for key in self.scalers_:
            columns = self.columns_scaler_ == key
            if columns.any():
                self.scalers_[key].fit(X[:, columns], y)

        return self

    def _transform(self, X, y=None):
        assert X.shape[1] == len(self.columns_scaler_)
        del y
        X = X.copy().astype(np.float32)
        for key in self.scalers_:
            columns = self.columns_scaler_ == key
            if columns.any():
                X[:, columns] = self.scalers_[key].transform(X[:, columns])
        return X


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
        cols = self.cols or X.columns.values
        for c in cols:
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

    def __init__(
        self,
        cols=None,
        return_df=True,
        handle_unknown="return_nan",
        folds=None,
    ):
        assert handle_unknown in ("error", "value", "return_nan")
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.handle_unknown = handle_unknown
        self.folds = folds
        self.train_df_ = None
        self.train_maps_ = None
        self.test_maps_ = None

    def fit(self, X, y):
        assert y is not None
        self.train_df_ = X
        self.train_maps_ = {}
        self.test_maps_ = {}
        cols = self.cols or X.columns.values
        prior = y.mean()
        # train用のencodingを作る
        if self.folds is None:
            # LOO
            for c in cols:
                targets = np.repeat(np.nan, len(X))
                for v in X[c].dropna().unique():
                    m = X[c] == v
                    m_size = m.sum()
                    if m_size >= 2:
                        y_m = y[m]
                        targets[m] = (y_m.sum() - y_m) / (m_size - 1)
                self.train_maps_[c] = targets
        else:
            # CV
            for fold, (train_indices, _) in enumerate(self.folds):
                X_t, y_t = X.iloc[train_indices], y[train_indices]
                for c in cols:
                    if c not in self.train_maps_:
                        self.train_maps_[c] = [{} for _ in range(len(self.folds))]
                    self.train_maps_[c][fold] = {}
                    values = X[c].dropna().unique()
                    for v in values:
                        m = X_t[c] == v
                        if m.any():
                            target = y_t[m].mean()
                        elif self.handle_unknown == "value":
                            target = prior
                        else:
                            target = np.nan
                        self.train_maps_[c][fold][v] = target
        # test用のencodingを作る
        for c in cols:
            self.test_maps_[c] = {}
            for v in X[c].dropna().unique():
                self.test_maps_[c][v] = y[X[c] == v].mean()
        return self

    def transform(self, X, y=None):
        del y
        X = X.copy()
        cols = self.cols or X.columns.values
        if X.equals(self.train_df_):
            if self.folds is None:
                for c in cols:
                    X[c] = self.train_maps_[c]
            else:
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
