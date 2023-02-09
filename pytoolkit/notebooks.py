"""JupyterLab向けのヘルパー関数など。"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display


def plot_histgrams(
    df_train: pl.DataFrame, df_test: pl.DataFrame, bins: int | str = "sturges"
):
    """ヒストグラムの描画ウィジェットを出力する。

    Examples:

        ::

            import pytoolkit.notebooks
            pytoolkit.notebooks.plot_histgrams(df_train, df_test)

    """

    floats_train = df_train.select(
        [pl.col(pl.Float32), pl.col(pl.Float64).cast(pl.Float32)]
    )
    floats_test = df_test.select(
        [pl.col(pl.Float32), pl.col(pl.Float64).cast(pl.Float32)]
    )
    # float_columns = list(set(floats_train.columns) & set(floats_test.columns))
    float_columns = [c for c in floats_train.columns if c in floats_test.columns]
    ignored_columns = [
        c
        for c in np.unique(list(df_train.columns) + list(df_test.columns))
        if c not in float_columns
    ]

    dropdown = widgets.Dropdown(options=float_columns, description="columns")
    dropdown2 = widgets.Dropdown(options=ignored_columns, description="ignored columns")
    output = widgets.Output()
    ax = plt.gca()  # get current axes

    def on_value_change(change) -> None:
        if change["name"] == "value":
            # old_value = change["old"]
            new_value = change["new"]
            ax.cla()
            train_values = df_train[new_value].drop_nulls().drop_nans().to_numpy()
            test_values = df_test[new_value].drop_nulls().drop_nans().to_numpy()
            assert len(train_values) > 0 or len(test_values) > 0, "データが空"
            sbins = np.histogram_bin_edges(
                np.concatenate([train_values, test_values]), bins=bins
            )
            train_hist, _ = np.histogram(train_values, bins=sbins)
            test_hist, _ = np.histogram(test_values, bins=sbins)
            train_hist = train_hist.astype(np.float32) / (train_hist.sum() + 1e-7)
            test_hist = test_hist.astype(np.float32) / (test_hist.sum() + 1e-7)
            ax.stairs(train_hist, sbins, fill=True, alpha=0.5, label="train")
            ax.stairs(test_hist, sbins, fill=True, alpha=0.5, label="test")
            ax.set_title(new_value)
            ax.legend()
            with output:
                output.clear_output(wait=True)
                display(ax.figure)

    plt.close()
    dropdown.observe(on_value_change)
    on_value_change({"name": "value", "old": float_columns[0], "new": float_columns[0]})
    display(dropdown, output, dropdown2)
