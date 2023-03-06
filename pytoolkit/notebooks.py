"""JupyterLab向けのヘルパー関数など。"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display


def display_numerics(
    df_train: pl.DataFrame, df_test: pl.DataFrame, bins: int | str = "sturges"
):
    """ヒストグラムの描画ウィジェットを出力する。

    Examples:

        ::

            import pytoolkit.notebooks
            pytoolkit.notebooks.display_numerics(df_train, df_test)

    """
    num_cols = [
        # https://pola-rs.github.io/polars-book/user-guide/datatypes.html
        pl.col(pl.Float32),
        pl.col(pl.Float64),
        pl.col(pl.Int8),
        pl.col(pl.Int16),
        pl.col(pl.Int32),
        pl.col(pl.Int64),
        pl.col(pl.UInt8),
        pl.col(pl.UInt16),
        pl.col(pl.UInt32),
        pl.col(pl.UInt64),
    ]

    nums_train = df_train.select(num_cols)
    nums_test = df_test.select(num_cols)
    # numeric_columns = list(set(nums_train.columns) & set(nums_test.columns))
    numeric_columns = [c for c in nums_train.columns if c in nums_test.columns]
    ignored_columns = [
        c
        for c in np.unique(list(df_train.columns) + list(df_test.columns))
        if c not in numeric_columns
    ]

    dropdown1 = widgets.Dropdown(options=numeric_columns, description="columns")
    dropdown2 = widgets.Dropdown(options=ignored_columns, description="ignored columns")
    output1 = widgets.Output()
    output2 = widgets.Output()
    output3 = widgets.Output()
    output4 = widgets.Output()
    all_widgets = [
        widgets.HBox([dropdown1, dropdown2]),
        output1,
        widgets.HBox(
            [
                widgets.VBox([widgets.Label("describe"), output2]),
                widgets.VBox([widgets.Label("train:value_counts"), output3]),
                widgets.VBox([widgets.Label("test:value_counts"), output4]),
            ]
        ),
    ]

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
            with output1:
                output1.clear_output(wait=True)
                display(ax.figure)
            with pl.Config() as cfg:
                cfg.set_tbl_cols(-1)
                cfg.set_tbl_rows(-1)
                with output2:
                    output2.clear_output(wait=True)
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
                cfg.set_tbl_rows(10)
                with output3:
                    output3.clear_output(wait=True)
                    display(
                        df_train[new_value]
                        .value_counts()
                        .sort("counts", descending=True)
                    )
                with output4:
                    output4.clear_output(wait=True)
                    display(
                        df_test[new_value]
                        .value_counts()
                        .sort("counts", descending=True)
                    )

    plt.close()
    dropdown1.observe(on_value_change)
    on_value_change(
        {"name": "value", "old": numeric_columns[0], "new": numeric_columns[0]}
    )
    display(*all_widgets)
