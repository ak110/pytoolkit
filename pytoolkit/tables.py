"""テーブルデータ関連。"""
import os
import pathlib

import numpy.typing as npt
import polars as pl


def load_labeled_data(
    data_path: str | os.PathLike[str], label_col_name: str
) -> tuple[pl.DataFrame, npt.NDArray]:
    """ラベルありデータの読み込み

    Args:
        data_path: データのパス(CSV, Excelなど)
        label_col_name: ラベルの列名

    Returns:
        データフレーム

    """
    data = load_unlabeled_data(data_path)
    labels = data.drop_in_place(label_col_name).to_numpy()
    return data, labels


def load_unlabeled_data(data_path: str | os.PathLike[str]) -> pl.DataFrame:
    """ラベルなしデータの読み込み

    Args:
        data_path: データのパス(CSV, Excelなど)

    Returns:
        データフレーム

    """
    data_path = pathlib.Path(data_path)
    data: pl.DataFrame
    if data_path.suffix.lower() == ".csv":
        data = pl.read_csv(data_path)
    elif data_path.suffix.lower() == ".tsv":
        data = pl.read_csv(data_path, sep="\t")
    elif data_path.suffix.lower() == ".arrow":
        data = pl.read_ipc(data_path)
    elif data_path.suffix.lower() == ".parquet":
        data = pl.read_parquet(data_path)
    elif data_path.suffix.lower() in (".xls", ".xlsx", ".xlsm"):
        data = pl.read_excel(data_path)  # type: ignore
    else:
        raise ValueError(f"Unknown suffix: {data_path}")
    return data
