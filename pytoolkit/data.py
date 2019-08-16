"""学習時のデータの読み込み周り。"""
import time

import numpy as np

import pytoolkit as tk

from . import keras


class Dataset:
    """訓練データなど。

    Args:
        data (ndarray or DataFrame): 入力データ
        labels (ndarray): ラベル
        groups (ndarray): グループID
        weights (ndarray): サンプルごとの重み
        ids (ndarray): ID (入力データにしないIDが別途必要な場合用)

    """

    def __init__(self, data, labels=None, groups=None, weights=None, ids=None):
        self.data = data
        self.labels = labels
        self.groups = groups
        self.weights = weights
        self.ids = ids

    def __len__(self):
        return len(self.data)

    def get_sample(self, index):
        """dataとlabelを返すだけの糖衣構文。"""
        if self.labels is None:
            return self.data[index], None
        return self.data[index], self.labels[index]

    def slice(self, rindex):
        """スライスを作成して返す。

        Args:
            rindex (array-like): インデックスの配列

        Returns:
            Dataset: スライス後のDataset

        """
        rindex = np.array(rindex)

        def _slice(d):
            if d is None:
                return None
            if hasattr(d, "iloc"):
                if rindex.dtype == bool:
                    return d.loc[rindex]
                else:
                    return d.iloc[rindex]
            return d[rindex]

        return Dataset(
            data=_slice(self.data),
            labels=_slice(self.labels),
            groups=_slice(self.groups),
            weights=_slice(self.weights),
            ids=_slice(self.ids),
        )

    def copy(self):
        """コピーの作成。

        Returns:
            Dataset: コピー

        """

        def _copy(d):
            if d is None:
                return None
            return d.copy()

        return Dataset(
            data=_copy(self.data),
            labels=_copy(self.labels),
            groups=_copy(self.groups),
            weights=_copy(self.weights),
            ids=_copy(self.ids),
        )


def concat(dataset1: Dataset, dataset2: Dataset) -> Dataset:
    """Dataset同士のconcat。"""
    import pandas as pd

    def _concat(a, b):
        if a is None or b is None:
            return None
        elif isinstance(a, pd.DataFrame):
            category_columns = a.select_dtypes("category").columns
            c = pd.concat([a, b], ignore_index=True, sort=False)
            if len(category_columns) > 0:
                # pd.concatでdtype=categoryが外れる列があるので再設定
                c[category_columns] = c[category_columns].astype("category")
            return c
        return np.concatenate([a, b], axis=0)

    return Dataset(
        data=_concat(dataset1.data, dataset2.data),
        labels=_concat(dataset1.labels, dataset2.labels),
        groups=_concat(dataset1.groups, dataset2.groups),
        weights=_concat(dataset1.weights, dataset2.weights),
        ids=_concat(dataset1.ids, dataset2.ids),
    )


def split(dataset: Dataset, count: int, shuffle=False):
    """Datasetを指定個数に分割する。"""
    dataset_size = len(dataset)
    sub_size = dataset_size // count
    assert sub_size > 0
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    return [
        dataset.slice(indices[o : o + sub_size])
        for o in range(0, dataset_size, sub_size)
    ]


class Preprocessor:
    """データ変換とかをするクラス。"""

    def get_sample(self, dataset, index):
        """datasetから1件のデータを取得。"""
        return dataset.get_sample(index)

    def collate(self, batch):
        """バッチサイズ分のデータを集約する処理。

        Args:
            batch (list): Datasetが返したデータをバッチサイズ分集めたもの。

        Returns:
            tuple: モデルに渡されるデータ。通常は入力データとラベルのtuple。

        """
        X_batch, y_batch = zip(*batch)

        # multiple input
        if isinstance(X_batch[0], list):
            X_batch = [
                np.array([x[i] for x in X_batch]) for i in range(len(X_batch[0]))
            ]
        else:
            X_batch = np.array(X_batch)

        # multiple output
        if isinstance(y_batch[0], list):
            y_batch = [
                np.array([y[i] for y in y_batch]) for i in range(len(y_batch[0]))
            ]
        else:
            y_batch = np.array(y_batch)

        return X_batch, y_batch


class DataLoader(keras.utils.Sequence):
    """データをバッチサイズずつ読み込むクラス。

    Args:
        dataset (Dataset): データセット。
        preprocessor (Preprocessor): データ変換とかをするクラス。
        batch_size (int): バッチサイズ。
        shuffle (bool): データをシャッフルするか否か。
        parallel (bool): 並列処理を行うか否か。
        use_horovod (bool): 1エポックあたりのミニバッチ数(__len__の戻り値)の算出にHorovodを考慮するか否か。

    Attributes:
        seconds_per_step (float): 1ステップ当たりの実処理時間の指数移動平均値。

    """

    def __init__(
        self,
        dataset,
        preprocessor=None,
        batch_size=32,
        shuffle=False,
        parallel=True,
        use_horovod=False,
    ):
        assert len(dataset) > 0
        self.dataset = dataset
        self.preprocessor = preprocessor or Preprocessor()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.parallel = parallel
        self.use_horovod = use_horovod
        self.seconds_per_step = 0

        if self.shuffle:
            # シャッフル時は常に同じバッチサイズを返せるようにする (学習時の安定性のため)
            mp_batch_size = (
                self.batch_size * tk.hvd.size() if self.use_horovod else self.batch_size
            )
            self.steps_per_epoch = -(-len(self.dataset) // mp_batch_size)  # ceiling
            self.index_generator = _generate_shuffled_indices(len(self.dataset))
            self.indices = [
                next(self.index_generator)
                for _ in range(self.steps_per_epoch * self.batch_size)
            ]
        else:
            # シャッフル無しの場合はそのまま順に返す。
            self.steps_per_epoch = -(-len(self.dataset) // self.batch_size)  # ceiling
            self.index_generator = None
            self.indices = np.arange(len(self.dataset))

    def on_epoch_end(self):
        """1エポック完了時の処理。(シャッフルする)"""
        if self.shuffle:
            self.indices = [
                next(self.index_generator)
                for _ in range(self.steps_per_epoch * self.batch_size)
            ]

    def __len__(self):
        """1エポックあたりのミニバッチ数を返す。"""
        return self.steps_per_epoch

    def __getitem__(self, index):
        """指定されたインデックスのミニバッチを1件返す。

        Args:
            index (int): データのインデックス。

        Returns:
            tuple: 入力データとラベル。

        """
        start_time = time.perf_counter()

        offset = self.batch_size * index
        batch_indices = self.indices[offset : offset + self.batch_size]
        assert not self.shuffle or len(batch_indices) == self.batch_size

        if self.parallel:
            futures = [
                tk.threading.get_pool().submit(self.get_sample, i)
                for i in batch_indices
            ]
            results = [f.result() for f in futures]
        else:
            results = [self.get_sample(i) for i in batch_indices]

        data = self.preprocessor.collate(results)

        elapsed_time = time.perf_counter() - start_time
        self.seconds_per_step = self.seconds_per_step * 0.99 + elapsed_time * 0.01

        return data

    def get_sample(self, index):
        """指定されたインデックスのデータを返す。

        Args:
            index (int): データのインデックス。

        Returns:
            tuple: 入力データとラベル。

        """
        return self.preprocessor.get_sample(self.dataset, index)

    def __iter__(self):
        """データを返す。"""
        for i in range(len(self)):
            yield self[i]


def _generate_shuffled_indices(data_count):
    """シャッフルしたインデックスのgenerator"""
    indices = np.arange(data_count)
    while True:
        np.random.shuffle(indices)
        yield from indices


def default_collate(batch):
    """バッチサイズ分のデータを集約する処理。

    Args:
        batch (list): Datasetが返したデータをバッチサイズ分集めたもの。

    Returns:
        tuple: モデルに渡されるデータ。通常は入力データとラベルのtuple。

    """
    X_batch, y_batch = zip(*batch)

    # multiple input
    if isinstance(X_batch[0], list):
        X_batch = [np.array([x[i] for x in X_batch]) for i in range(len(X_batch[0]))]
    else:
        X_batch = np.array(X_batch)

    # multiple output
    if isinstance(y_batch[0], list):
        y_batch = [np.array([y[i] for y in y_batch]) for i in range(len(y_batch[0]))]
    else:
        y_batch = np.array(y_batch)

    return X_batch, y_batch
