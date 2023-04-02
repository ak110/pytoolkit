"""テーブルデータ関連。"""
import logging
import os
import pathlib

import numpy as np
import numpy.typing as npt
import polars as pl

logger = logging.getLogger(__name__)


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
        data = pl.read_csv(data_path, separator="\t")
    elif data_path.suffix.lower() == ".arrow":
        data = pl.read_ipc(data_path)
    elif data_path.suffix.lower() == ".parquet":
        data = pl.read_parquet(data_path)
    elif data_path.suffix.lower() in (".xls", ".xlsx", ".xlsm"):
        data = pl.read_excel(data_path)  # type: ignore
    else:
        raise ValueError(f"Unknown suffix: {data_path}")
    return data


def remove_correlated(df: pl.DataFrame, threshold: float = 0.9) -> pl.DataFrame:
    """相関係数の高い列を削除。"""
    return df.select(pl.exclude(detect_correlated(df, threshold)))


def detect_correlated(df: pl.DataFrame, threshold: float = 0.9) -> list[str]:
    """相関係数の高い列をリストアップ。"""
    df = df.fill_nan(None).fill_null(strategy="mean")
    removed: list[str] = []
    while True:
        corr = df.corr()
        cond = corr.select(pl.all().abs() >= threshold).to_numpy() & ~np.eye(
            corr.shape[0], dtype=np.bool_
        )
        scores = cond.sum(axis=0)
        i = np.argmax(scores)
        if scores[i] <= 0:
            break
        logger.info(f"detect_correlated: {corr.columns[i]} score={scores[i]}")
        removed.append(corr.columns[i])
        df.drop_in_place(corr.columns[i])
    return removed
