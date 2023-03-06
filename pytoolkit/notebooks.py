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

    nums_train = df_train.select(
        [
            pl.col(pl.Float32),
            pl.col(pl.Float64).cast(pl.Float32),
            pl.col(pl.Int32),
            pl.col(pl.Int64),
        ]
    )
    nums_test = df_test.select(
        [
            pl.col(pl.Float32),
            pl.col(pl.Float64).cast(pl.Float32),
            pl.col(pl.Int32),
            pl.col(pl.Int64),
        ]
    )
    # numeric_columns = list(set(nums_train.columns) & set(nums_test.columns))
    numeric_columns = [c for c in nums_train.columns if c in nums_test.columns]
    ignored_columns = [
        c
        for c in np.unique(list(df_train.columns) + list(df_test.columns))
        if c not in numeric_columns
    ]

    dropdown = widgets.Dropdown(options=numeric_columns, description="columns")
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
                display(
                    pl.concat(
                        [
                            df_train[new_value]
                            .describe()
                            .select(
                                pl.col("statistic"), pl.col("value").alias("train")
                            ),
                            df_test[new_value]
                            .describe()
                            .select(pl.col("value").alias("test")),
                        ],
                        how="horizontal",
                    )
                )
                display(
                    df_train[new_value].value_counts().sort("counts", descending=True),
                    df_test[new_value].value_counts().sort("counts", descending=True),
                )

    plt.close()
    dropdown.observe(on_value_change)
    on_value_change(
        {"name": "value", "old": numeric_columns[0], "new": numeric_columns[0]}
    )
    display(dropdown, output, dropdown2)
