"""学習時のデータの読み込み周り。"""
import abc
import time

import numpy as np

import pytoolkit as tk

from . import keras


class Dataset(metaclass=abc.ABCMeta):
    """データを読み込むクラス。"""

    @abc.abstractmethod
    def __len__(self):
        """データ件数を返す。"""
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index):
        """指定されたインデックスのデータを返す。

        Args:
            index (int): データのインデックス。

        Returns:
            tuple: Preprocessorに渡されるデータ。通常は入力データとラベルのtuple。

        """
        raise NotImplementedError

    def __iter__(self):
        """データを返す。"""
        for i in range(len(self)):
            yield self[i]


class TupleDataset(Dataset):
    """タプルを持つデータセット"""

    def __init__(self, *datasets):
        assert all([len(d) == len(datasets[0]) for d in datasets])
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        return tuple([d[index] for d in self.datasets])


class DataLoader(keras.utils.Sequence):
    """データをバッチサイズずつ読み込むクラス。

    Args:
        dataset (Dataset): データセット。
        batch_size (int): バッチサイズ。
        shuffle (bool): データをシャッフルするか否か。
        parallel (bool): 並列処理を行うか否か。
        use_horovod (bool): 1エポックあたりのミニバッチ数(__len__の戻り値)の算出にHorovodを考慮するか否か。
        collate_fn (callable): 集約処理

    Attributes:
        seconds_per_step (float): 1ステップ当たりの実処理時間の指数移動平均値。

    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        parallel=True,
        use_horovod=False,
        collate_fn=None,
    ):
        assert len(dataset) > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.parallel = parallel
        self.use_horovod = use_horovod
        self.collate_fn = collate_fn or default_collate
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

        data = self.collate_fn(results)

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
        return self.dataset[index]

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


class SubDataset(Dataset):
    """Datasetの一部に対するDataset。

    Args:
        dataset (Dataset): 元のDataset。
        indices (list-like of int): インデックスの配列。

    """

    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def split(dataset, count, shuffle=False):
    """Datasetを指定個数に分割する。"""
    dataset_size = len(dataset)
    sub_size = dataset_size // count
    assert sub_size > 0
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    return [
        SubDataset(dataset, indices[o : o + sub_size])
        for o in range(0, dataset_size, sub_size)
    ]
