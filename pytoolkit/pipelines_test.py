"""テストコード"""

import polars as pl

import pytoolkit as tk


class InputStep(tk.pipelines.Step):
    def run(self, df: pl.DataFrame, run_type: tk.pipelines.RunType) -> pl.DataFrame:
        """当該ステップの処理。Pipeline経由で呼び出される。"""
        return pl.DataFrame({"run_type": [run_type] * 3})


def test_step(tmpdir):
    run_type: tk.pipelines.RunType = "train"
    df = tk.pipelines.Pipeline(tmpdir / "models", tmpdir / "cache").run(
        InputStep, run_type
    )
    assert tuple(df["run_type"].to_list()) == tuple([run_type] * 3)