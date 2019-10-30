import numpy as np
import pandas as pd
import pytest

import pytoolkit as tk


@pytest.mark.parametrize("category", ["category", "ordinal", "onehot"])
def test_features_encoder(category):
    train_set = tk.datasets.load_boston()
    df_train = train_set.data
    df_train["CHAS"] = df_train["CHAS"].astype("bool")
    df_train["RAD"] = df_train["RAD"].astype("category")  # 適切ではないがテストのための変換
    df_train["TAX"] = df_train["TAX"].astype("object")  # 適切ではないがテストのための変換
    df_test = df_train.copy()

    # 列の順番が変わっても結果が同じになる事を確認する
    df_test = df_test[np.random.permutation(df_test.columns.values)]

    encoder = tk.preprocessing.FeaturesEncoder(category=category)
    df_train = encoder.fit_transform(df_train, train_set.labels)
    df_test = encoder.transform(df_test)
    assert df_train.equals(df_test)


def test_target_encoder():
    df = pd.DataFrame()
    df["a"] = ["a", "b", "a", "b"]
    df["b"] = [0, 1, 0, 1]
    df["c"] = [0.0, 1.0, 0.0, np.nan]
    df["d"] = [0, 1, 0, 1]
    df["b"] = df["b"].astype("category")
    y = np.array([1, 3, 5, 7])

    encoder = tk.preprocessing.TargetEncoder(cols=["a", "b", "c"], min_samples_leaf=1)
    encoder.fit(df, y)
    df2 = encoder.transform(df)

    assert df2["a"].values == pytest.approx([1.0, 2.0, 1.0, 2.0])
    assert df2["b"].values == pytest.approx([1.0, 2.0, 1.0, 2.0])
    assert df2["c"].values == pytest.approx([1.0, 2.0, 1.0, np.nan], nan_ok=True)
    assert df2["d"].values == pytest.approx([0, 1, 0, 1])


def test_normalizer():
    df = pd.DataFrame()
    df["a"] = [1, 2, 3, 4]
    df["b"] = [0, 1, 0, 1]
    df["c"] = [0.0, 1.0, 0.0, np.nan]
    df["d"] = [1, 1, None, 1]

    r1 = tk.preprocessing.Normalizer().fit_transform(df)
    r2 = tk.preprocessing.Normalizer().fit_transform(df.values)

    assert r1["a"].values == pytest.approx(r2[:, 0])
    assert r1["b"].values == pytest.approx(r2[:, 1])
    assert r1["c"].values == pytest.approx(r2[:, 2], nan_ok=True)
    assert r1["d"].values == pytest.approx(r2[:, 3], nan_ok=True)

    assert r1["a"].values == pytest.approx(
        [-1.3416407, -0.4472136, 0.4472136, 1.3416407]
    )
    assert r1["b"].values == pytest.approx([-1, 1, -1, 1])
    assert r1["c"].values == pytest.approx([0, 2.1213205, 0, np.nan], nan_ok=True)
    assert r1["d"].values == pytest.approx([0, 0, np.nan, 0], nan_ok=True)
