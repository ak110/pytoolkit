"""画像分類関連。"""
from __future__ import annotations

import pathlib
import typing
import xml.etree.ElementTree

import numpy as np
import pandas as pd

import pytoolkit as tk


def load_image_folder(
    data_dir: tk.typing.PathLike,
    class_names: typing.Sequence[str] = None,
    use_tqdm: bool = True,
    check_image: bool = False,
) -> tk.data.Dataset:
    """画像分類でよくある、クラス名でディレクトリが作られた階層構造のデータ。

    Args:
        data_dir: 対象ディレクトリ
        class_names: クラス名の配列
        use_tqdm: tqdmを使用するか否か
        check_image: 画像として読み込みチェックを行い、読み込み可能なファイルのみ返すか否か (遅いので注意)

    Returns:
        Dataset。metadata['class_names']にクラス名の配列。

    """
    class_names, X, y = tk.ml.listup_classification(
        data_dir, class_names=class_names, use_tqdm=use_tqdm, check_image=check_image
    )
    return tk.data.Dataset(X, y, metadata={"class_names": class_names})


def load_trainval_folders(data_dir, swap=False):
    """data_dir直下のtrainとvalをload_image_folderで読み込む。"""
    data_dir = pathlib.Path(data_dir)
    train_set = load_image_folder(data_dir / "train")
    val_set = load_image_folder(data_dir / "val", train_set.metadata.get("class_names"))
    if swap:
        train_set, val_set = val_set, train_set
    return train_set, val_set


def load_train1000():
    """train with 1000なデータの読み込み。

    References:
        - <https://github.com/mastnk/train1000>

    """
    train_set, test_set = tk.datasets.load_cifar10()
    train_set = extract_class_balanced(train_set, num_classes=10, samples_per_class=100)
    return train_set, test_set


def load_imagenet(data_dir: tk.typing.PathLike, verbose: bool = True):
    """ImageNet (ILSVRC 2012のClassification)のデータの読み込み。

    Args:
        data_dir: ディレクトリ。(Annotations, Data, ImageSetが入っているところ)
        verbose: 読み込み状況をtqdmで表示するならTrue

    """
    data_dir = pathlib.Path(data_dir)

    class_index_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    df = pd.read_json(class_index_url).T
    class_names = df[0].values
    class_names_to_id = dict(zip(class_names, df[0].index))

    train_dir = data_dir / "Data/CLS-LOC/train"
    train_set = load_image_folder(train_dir, class_names=class_names)

    X_val, y_val = [], []
    val_dir = data_dir / "Annotations/CLS-LOC/val"
    for xml_path in tk.utils.tqdm(list(val_dir.glob("*.xml")), disable=not verbose):
        root = xml.etree.ElementTree.parse(str(xml_path)).getroot()
        class_name: str = root.find("object").find("name").text  # type: ignore

        X_val.append(data_dir / f"Data/CLS-LOC/val/{xml_path.stem}.JPEG")
        y_val.append(class_names_to_id[class_name])
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    val_set = tk.data.Dataset(X_val, y_val, metadata={"class_names": class_names})
    return train_set, val_set


def extract_class_balanced(dataset, num_classes, samples_per_class):
    """クラスごとに均等に抜き出す。dataset.labelsは[0, num_classes)の値のndarrayを前提とする。"""
    index_list: list[np.intp] = []
    for c in range(num_classes):
        index_list.extend(np.where(dataset.labels == c)[0][:samples_per_class])
    return dataset.slice(index_list)
