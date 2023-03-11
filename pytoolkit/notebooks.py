"""JupyterLab向けのヘルパー関数など。"""

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import display


def describe(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    bins: int | str = "sturges",
    labels: tuple[str, str] = ("train", "test"),
):
    """ヒストグラムの描画ウィジェットを出力する。

    Args:
        bins: ヒストグラムのbin数の指定。
              <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>
        labels: 1個目と2個目のdfの表示名。(既定値は("train", "test"))

    Examples:

        ::

            import pytoolkit.notebooks
            pytoolkit.notebooks.describe(df_train, df_test)

    """
    # 両方に存在する列が対象
    describe_columns = [c for c in df_train.columns if c in df_test.columns]
    # 対象外の列
    ignored_columns = [
        c
        for c in np.unique(list(df_train.columns) + list(df_test.columns))
        if c not in describe_columns
    ]

    dropdown1 = widgets.Dropdown(options=describe_columns, description="columns")
    dropdown2 = widgets.Dropdown(options=ignored_columns, description="ignored")
    output1 = widgets.Output()
    output2 = widgets.Output()
    output3 = widgets.Output()
    output4 = widgets.Output()
    all_widgets = [
        widgets.HBox([dropdown1, dropdown2]) if len(ignored_columns) > 0 else dropdown1,
        output1,
        widgets.HBox(
            [
                widgets.VBox([widgets.Label("describe"), output2]),
                widgets.VBox([widgets.Label(labels[0] + ":value_counts"), output3]),
                widgets.VBox([widgets.Label(labels[1] + ":value_counts"), output4]),
            ]
        ),
    ]

    ax = plt.gca()  # get current axes

    def on_value_change(change) -> None:
        if change["name"] != "value":
            return
        # old_value = change["old"]
        new_value = change["new"]

        with output1:
            output1.clear_output(wait=True)
            ax.cla()
            if df_train[new_value].dtype in (
                # https://pola-rs.github.io/polars-book/user-guide/datatypes.html
                pl.Float32,
                pl.Float64,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ):
                _plot_hist(new_value)
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
                                pl.col("statistic"), pl.col("value").alias(labels[0])
                            ),
                            df_test[new_value]
                            .describe()
                            .select(pl.col("value").alias(labels[1])),
                        ],
                        how="horizontal",
                    )
                )
            cfg.set_tbl_rows(10)
            with output3:
                output3.clear_output(wait=True)
                display(
                    df_train[new_value].value_counts().sort("counts", descending=True)
                )
            with output4:
                output4.clear_output(wait=True)
                display(
                    df_test[new_value].value_counts().sort("counts", descending=True)
                )

    def _plot_hist(new_value):
        """ヒストグラムの描画"""
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
        ax.stairs(train_hist, sbins, fill=True, alpha=0.5, label=labels[0])
        ax.stairs(test_hist, sbins, fill=True, alpha=0.5, label=labels[1])
        ax.set_title(new_value)
        ax.legend()

    plt.close()
    dropdown1.observe(on_value_change)
    on_value_change(
        {"name": "value", "old": describe_columns[0], "new": describe_columns[0]}
    )
    display(*all_widgets)
