"""前処理関連。"""
from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import sklearn.base
import sklearn.pipeline
import sklearn.preprocessing

import pytoolkit as tk


def encode_binary(s: pd.Series, true_value, false_value) -> pd.Series:
    """列の2値化。"""
    s2 = s.map({true_value: True, false_value: False})
    assert s2.notnull().all(), f"Convert error: {s[s2.isnull()].value_counts()}"
    return s2.astype(bool)


def encode_ordinal(s: pd.Series, values) -> pd.Series:
    """順序のあるカテゴリ変数の数値化。"""
    mask = s.notnull()
    result = s.map({v: i for i, v in enumerate(values)})
    # 未知の値があればエラー
    if result[mask].isnull().any():
        raise ValueError(f"Unknown values: {set(s.unique()) - set(values)}")
    # nullが無ければメモリ節約のため(?)int32に変換
    if s.notnull().all():
        result = result.astype(np.int32)
    return result


def encode_cyclic(
    s: pd.Series, min_value: float = 0, max_value: float = 1
) -> pd.DataFrame:
    """周期性のある数値のsin/cos化。"""
    rad = 2 * np.pi * (s - min_value) / (max_value - min_value)
    df = pd.DataFrame()
    df["sin"] = np.sin(rad)
    df["cos"] = np.cos(rad)
    return df


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


class FeaturesEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """特徴を学習器に与えられるように色々いい感じに変換するクラス。

    Args:
        category: カテゴリ変数の処理方法
            - "category": Ordinalエンコードしてcategory型にする (LightGBMとか用)
            - "ordinal": Ordinalエンコードしてobject型にする (XGBoost/CatBoostとか用)
            - "onehot": One-hotエンコードしてbool型にする (NNとか用)
        binary_fraction: 0-1
        iszero_fraction: 0-1
        isnull_fraction: 0-1
        ordinal_encoder: OrdinalEncoderのインスタンス
        onehot_encoder: OneHotEncoderのインスタンス
        count_encoder: CountEncoderのインスタンス
        target_encoder: TargetEncoderのインスタンス
        ignore_cols: 無視する列名

    """

    def __init__(
        self,
        category: str = "category",
        binary_fraction: float = 0.01,
        iszero_fraction: float = 0.01,
        isnull_fraction: float = 0.01,
        rare_category_fraction: float = 0.01,
        ordinal_encoder: sklearn.base.BaseEstimator = None,
        onehot_encoder: sklearn.base.BaseEstimator = None,
        count_encoder: sklearn.base.BaseEstimator = None,
        target_encoder: sklearn.base.BaseEstimator = None,
        ignore_cols: typing.Sequence[str] = None,
    ):
        import category_encoders as ce

        assert category in ("category", "ordinal", "onehot")
        self.category = category
        self.binary_fraction = binary_fraction
        self.iszero_fraction = iszero_fraction
        self.isnull_fraction = isnull_fraction
        self.rare_category_fraction = rare_category_fraction
        self.ordinal_encoder = ordinal_encoder or OrdinalEncoder()
        self.onehot_encoder = onehot_encoder or ce.OneHotEncoder(use_cat_names=True)
        self.count_encoder = count_encoder or CountEncoder()
        self.target_encoder = target_encoder or TargetEncoder()
        self.ignore_cols = ignore_cols or []
        self.binary_cols_: typing.Optional[typing.List[str]] = None
        self.numeric_cols_: typing.Optional[typing.List[str]] = None
        self.category_cols_: typing.Optional[typing.List[str]] = None
        self.rare_category_cols_: typing.Optional[typing.List[str]] = None
        self.iszero_cols_: typing.Optional[typing.List[str]] = None
        self.isnull_cols_: typing.Optional[typing.List[str]] = None
        self.feature_names_: typing.Optional[typing.List[str]] = None
        self.notnull_cols_: typing.Optional[pd.Series] = None

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        del fit_params
        with tk.log.trace("FeaturesEncoder.fit_transform"):
            # 対象とする列のリストアップ
            candidate_columns = set(X.columns.values)
            # ignore_colsを除外
            candidate_columns -= set(self.ignore_cols)
            # 値が1種類の列を削除 (Null + 1種類の値だと消えちゃうので注意: 先にisnull列を作っておけばOK)
            candidate_columns -= set(X.nunique()[lambda s: s <= 1].index)

            # 型で振り分ける
            cols = tk.table.group_columns(X, sorted(candidate_columns))
            assert len(cols["unknown"]) == 0, f"Unknown dtypes: {cols['unknown']}"

            # 以下、列ごとの処理

            self.binary_cols_ = [
                c
                for c in cols["binary"]
                if self.binary_fraction <= X[c].mean() <= (1 - self.binary_fraction)
            ]

            self.numeric_cols_ = cols["numeric"]
            self.category_cols_ = cols["categorical"]
            assert self.numeric_cols_ is not None
            assert self.category_cols_ is not None

            if len(self.category_cols_) > 0:
                if self.category in ("category", "ordinal"):
                    self.ordinal_encoder.fit(X[self.category_cols_].astype(object))
                elif self.category in ("onehot",):
                    self.onehot_encoder.fit(X[self.category_cols_].astype(object))

            self.rare_category_cols_ = [
                c
                for c in self.category_cols_
                if X[c].nunique() >= 3
                and X[c].value_counts().min() <= len(X) * self.rare_category_fraction
            ]
            if len(self.rare_category_cols_) > 0:
                self.target_encoder.fit(X[self.rare_category_cols_], y)
                self.count_encoder.fit(X[self.rare_category_cols_], y)

            self.iszero_cols_ = [
                c
                for c in self.numeric_cols_
                if self.iszero_fraction
                <= (X[c] == 0).mean()
                <= (1 - self.iszero_fraction)
            ]
            self.isnull_cols_ = [
                c
                for c in self.numeric_cols_ + self.category_cols_
                if self.isnull_fraction
                <= X[c].isnull().mean()
                <= (1 - self.isnull_fraction)
            ]

            self.feature_names_ = None
            self.notnull_cols_ = None
            feats = self.transform(X, y)
            # 値の重複列の削除
            self.feature_names_ = feats.columns.values[~feats.T.duplicated()]
            feats = feats[self.feature_names_]
            # trainにnullが含まれない列を覚えておく (チェック用)
            self.notnull_cols_ = feats.notnull().all(axis=0)

            return feats

    def transform(self, X, y=None):
        assert self.binary_cols_ is not None
        assert self.numeric_cols_ is not None
        assert self.category_cols_ is not None
        assert self.rare_category_cols_ is not None
        assert self.iszero_cols_ is not None
        assert self.isnull_cols_ is not None
        del y
        with tk.log.trace("FeaturesEncoder.transform"):
            feats = pd.DataFrame(index=X.index)

            if len(self.binary_cols_) > 0:
                feats[self.binary_cols_] = X[self.binary_cols_].astype(np.bool)

            if len(self.numeric_cols_) > 0:
                feats[self.numeric_cols_] = X[self.numeric_cols_].astype(np.float32)

            if len(self.category_cols_) > 0:
                if self.category == "category":
                    feats[self.category_cols_] = self.ordinal_encoder.transform(
                        X[self.category_cols_].astype(object)
                    ).astype("category")
                elif self.category in "ordinal":
                    feats[self.category_cols_] = (
                        self.ordinal_encoder.transform(
                            X[self.category_cols_].astype(object)
                        )
                        .astype("object")
                        .fillna(-1)
                    )
                elif self.category in ("onehot",):
                    fn = self.onehot_encoder.get_feature_names()
                    tk.table.add_cols(
                        feats,
                        fn,
                        self.onehot_encoder.transform(
                            X[self.category_cols_].astype(object)
                        )[fn],
                    )

            if len(self.rare_category_cols_) > 0:
                tk.table.add_cols(
                    feats,
                    [f"{c}_target" for c in self.rare_category_cols_],
                    self.target_encoder.transform(X[self.rare_category_cols_]),
                )
                tk.table.add_cols(
                    feats,
                    [f"{c}_count" for c in self.rare_category_cols_],
                    self.count_encoder.transform(X[self.rare_category_cols_]),
                )

            if len(self.iszero_cols_) > 0:
                tk.table.add_cols(
                    feats,
                    [f"{c}_iszero" for c in self.iszero_cols_],
                    X[self.iszero_cols_] == 0,
                )
            if len(self.isnull_cols_) > 0:
                tk.table.add_cols(
                    feats,
                    [f"{c}_isnull" for c in self.isnull_cols_],
                    X[self.isnull_cols_].isnull(),
                )

            # infチェック
            isinf_cols = np.isinf(feats.astype(np.float32)).any(axis=0)
            if isinf_cols.any():
                raise RuntimeError(f"inf: {feats.columns.values[isinf_cols]}")

            # 値の重複列の削除
            if self.feature_names_ is not None:
                feats = feats[self.feature_names_]
            # testにのみnullが含まれないかチェック
            if self.notnull_cols_ is not None:
                test_only_null = feats.isnull().any(axis=0) & self.notnull_cols_
                if test_only_null.any():
                    raise RuntimeError(
                        f"test only null: {test_only_null[test_only_null].index.values}"
                    )

            return feats


class OrdinalEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Ordinal Encoding。

    Category Encoders <http://contrib.scikit-learn.org/categorical-encoding/> 風のインターフェースにしてみる。
    (全部は作っていない)

    nanはfit時に存在するならvalue扱いにして、test時にのみ出てきたらerrorにする。
    fit時に出てこなかったカテゴリもerrorにする。

    """

    def __init__(self, cols=None, return_df=True, output_dtype="category"):
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.output_dtype = output_dtype
        self.maps_ = None

    def fit(self, X, y=None):
        del y
        self.maps_ = {}
        for c in self.cols or X.columns.values:
            values = X[c].unique()
            self.maps_[c] = dict(zip(values, range(len(values))))
        return self

    def transform(self, X, y=None):
        del y
        X = X.copy()
        assert self.maps_ is not None
        for c in self.cols or X.columns.values:
            s = X[c].map(self.maps_[c])
            if s.isnull().any():
                unk = set(X[c].unique()) - set(self.maps_[c])
                raise ValueError(f"Unknown values in column '{c}': {unk}")
            X[c] = s.astype(self.output_dtype)
        return X if self.return_df else X.values


class CountEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Count Encoding。

    Category Encoders <http://contrib.scikit-learn.org/categorical-encoding/> 風のインターフェースにしてみる。
    (全部は作っていない)

    """

    def __init__(self, cols=None, return_df=True, handle_missing="return_nan"):
        assert handle_missing in ("error", "value", "return_nan")
        super().__init__()
        self.cols = cols
        self.return_df = return_df
        self.handle_missing = handle_missing
        self.maps_ = None

    def fit(self, X, y=None):
        del y
        self.maps_ = {}
        for c in self.cols or X.columns.values:
            self.maps_[c] = X[c].value_counts(dropna=False).astype(np.float32).to_dict()
            if np.nan in self.maps_[c]:
                if self.handle_missing == "error":
                    raise ValueError(f"Column '{c}' has NaN!")
                if self.handle_missing == "return_nan":
                    self.maps_[c][np.nan] = np.nan
        return self

    def transform(self, X, y=None):
        del y
        X = X.copy()
        assert self.maps_ is not None
        for c in self.cols or X.columns.values:
            X[c] = X[c].map(self.maps_[c]).astype(np.float32)
        return X if self.return_df else X.values


class TargetEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Target Encoding。

    foldを切った方が少し安全という話はあるが、
    trainとtestで傾向が変わりかねなくてちょっと嫌なのでfold切らないものを自作した。

    お気持ちレベルだけどtargetそのままじゃなくrank化するようにしてみた(order=True)り、
    min_samples_leaf未満のカテゴリはnp.nanになるようにしたり。

    """

    def __init__(self, cols=None, return_df=True, min_samples_leaf=3, order=True):
        super().__init__()
        assert min_samples_leaf >= 1
        self.cols = cols
        self.return_df = return_df
        self.min_samples_leaf = min_samples_leaf
        self.order = order
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
        if self.order:
            s = s.argsort() + 1
        return s.to_dict()

    def transform(self, X, y=None):
        del y
        X = X.copy()
        assert self.maps_ is not None
        for c in self.cols or X.columns.values:
            X[c] = X[c].map(self.maps_[c]).astype(np.float32)
        return X if self.return_df else X.values


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
            assert not np.isinf(values).any(), f"Invalid value: {c=} {values=}"

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
        assert self.params_ is not None
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
