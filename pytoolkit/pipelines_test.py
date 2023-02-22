"""テストコード"""

import polars as pl

import pytoolkit.pipelines


class InputStep(pytoolkit.pipelines.Step):
    def __init__(self) -> None:
        super().__init__()
        self.name = "input_step"

    def run(
        self, df: pl.DataFrame, run_type: pytoolkit.pipelines.RunType
    ) -> pl.DataFrame:
        """当該ステップの処理。Pipeline経由で呼び出される。"""
        return pl.DataFrame({"run_type": [run_type] * 3})


def test_step(tmpdir):
    run_type: pytoolkit.pipelines.RunType = "train"
    pipeline = pytoolkit.pipelines.Pipeline(tmpdir / "models", tmpdir / "cache")
    pipeline.add(InputStep())
    df = pipeline.invoke("input_step", run_type)
    assert tuple(df["run_type"].to_list()) == tuple([run_type] * 3)
