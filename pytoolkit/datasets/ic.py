"""画像分類関連。"""
import pathlib

import numpy as np

import pytoolkit as tk


def load_image_folder(data_dir, class_names=None, use_tqdm=True, check_image=False):
    """画像分類でよくある、クラス名でディレクトリが作られた階層構造のデータ。

    Args:
        data_dir (PathLike): 対象ディレクトリ
        class_names (ArrayLike): クラス名の配列
        use_tqdm (bool): tqdmを使用するか否か
        check_image (bool): 画像として読み込みチェックを行い、読み込み可能なファイルのみ返すか否か (遅いので注意)

    Returns:
        tk.data.Dataset: Dataset。metadata['class_names']にクラス名の配列。

    """
    class_names, X, y = tk.ml.listup_classification(
        data_dir, class_names=class_names, use_tqdm=use_tqdm, check_image=check_image
    )
    return tk.data.Dataset(X, y, metadata={"class_names": class_names})


def load_train_val_image_folders(data_dir, swap=False):
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


def extract_class_balanced(dataset, num_classes, samples_per_class):
    """クラスごとに均等に抜き出す。dataset.labelsは[0, num_classes)の値のndarrayを前提とする。"""
    index_list = []
    for c in range(num_classes):
        index_list.extend(np.where(dataset.labels == c)[0][:samples_per_class])
    return dataset.slice(index_list)
