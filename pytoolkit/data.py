"""学習時のデータの読み込み周り。

dataset: dataの集合
data: 1件のデータ
sample: 学習時に使用する1件のデータ (1件以上のdataから作られるもの) (という事にしている)
batch: sampleのバッチサイズ個の集合

"""
from __future__ import annotations

import abc
import dataclasses
import random
import time
import typing

import numpy as np
import pandas as pd
import tensorflow as tf

import pytoolkit as tk


@dataclasses.dataclass()
class Dataset:
    """訓練データなど。

    Args:
        data: 入力データ
        labels: ラベル
        groups: グループID
        weights: サンプルごとの重み
        ids: ID (入力データにしないIDが別途必要な場合用)
        init_score: LightGBMなど用。boostingのベーススコア。
        metadata: メタデータ。色々独自に入れておいてOK。sliceとかではそのままコピーされたりするので注意。

    """

    data: typing.Union[typing.Sequence[typing.Any], pd.DataFrame, dict]
    labels: np.ndarray = None
    groups: np.ndarray = None
    weights: np.ndarray = None
    ids: np.ndarray = None
    init_score: np.ndarray = None
    metadata: typing.Optional[dict] = None

    def __len__(self) -> int:
        """データ件数を返す。"""
        return len(self.data)

    def get_data(self, index: int) -> typing.Tuple[typing.Any, typing.Any]:
        """dataとlabelを返すだけの糖衣構文。"""
        if self.labels is None:
            return self.data[index], None
        self.labels = typing.cast(np.ndarray, self.labels)
        return self.data[index], self.labels[index]

    def slice(self, rindex: typing.Sequence[int]) -> Dataset:
        """スライスを作成して返す。

        Args:
            rindex: インデックスの配列

        Returns:
            スライス後のDataset

        """
        rindex = np.array(rindex)
        return self.__class__(
            data=self.__class__.slice_field(self.data, rindex),
            labels=self.__class__.slice_field(self.labels, rindex),
            groups=self.__class__.slice_field(self.groups, rindex),
            weights=self.__class__.slice_field(self.weights, rindex),
            ids=self.__class__.slice_field(self.ids, rindex),
            init_score=self.__class__.slice_field(self.init_score, rindex),
            metadata=self.metadata.copy() if self.metadata is not None else None,
        )

    def copy(self) -> Dataset:
        """コピーを作成して返す。

        Returns:
            コピー

        """
        return self.__class__(
            data=self.__class__.copy_field(self.data),
            labels=self.__class__.copy_field(self.labels),
            groups=self.__class__.copy_field(self.groups),
            weights=self.__class__.copy_field(self.weights),
            ids=self.__class__.copy_field(self.ids),
            init_score=self.__class__.copy_field(self.init_score),
            metadata=self.metadata.copy() if self.metadata is not None else None,
        )

    @classmethod
    def concat(cls, dataset1, dataset2) -> Dataset:
        """Dataset同士のconcat。"""
        return cls(
            data=cls.concat_field(dataset1.data, dataset2.data),
            labels=cls.concat_field(dataset1.labels, dataset2.labels),
            groups=cls.concat_field(dataset1.groups, dataset2.groups),
            weights=cls.concat_field(dataset1.weights, dataset2.weights),
            ids=cls.concat_field(dataset1.ids, dataset2.ids),
            init_score=cls.concat_field(dataset1.init_score, dataset2.init_score),
            metadata=dataset1.metadata.copy()
            if dataset1.metadata is not None
            else None,
        )

    @classmethod
    def slice_field(cls, d, rindex: typing.Sequence[int]):
        """値のスライスを作成して返す。"""
        if d is None:
            return None
        if isinstance(d, dict):
            return {k: cls.slice_field(v, rindex) for k, v in d.items()}
        elif hasattr(d, "iloc"):
            if np.asarray(rindex).dtype == bool:
                return d.loc[rindex]
            else:
                return d.iloc[rindex]
        return d[rindex]

    @classmethod
    def copy_field(cls, d):
        """値のコピーを作成して返す。"""
        if d is None:
            return None
        elif isinstance(d, dict):
            return {k: cls.copy_field(v) for k, v in d.items()}
        assert isinstance(d, (list, np.ndarray, pd.Series, pd.DataFrame))
        return d.copy()

    @classmethod
    def concat_field(cls, a, b):
        """2個の値のconcat。"""
        if a is None:
            assert b is None
            return None
        elif isinstance(a, dict):
            assert isinstance(b, dict)
            assert tuple(a) == tuple(b)
            return {k: cls.concat_field(a[k], b[k]) for k in a}
        elif isinstance(a, pd.DataFrame):
            assert isinstance(b, pd.DataFrame)
            category_columns = a.select_dtypes("category").columns
            c = pd.concat([a, b], ignore_index=True, sort=False)
            if len(category_columns) > 0:
                # pd.concatでdtype=categoryが外れる列があるので再設定
                c[category_columns] = c[category_columns].astype("category")
            return c
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        return np.concatenate([a, b], axis=0)


def split(dataset: Dataset, count: int, shuffle=False):
    """Datasetを指定個数に分割する。"""
    dataset_size = len(dataset)
    sub_size = -(-dataset_size // count)  # 端数切り上げ
    assert sub_size > 0
    indices = np.arange(dataset_size)
    if shuffle:
        random.shuffle(indices)
    return [
        dataset.slice(indices[o : o + sub_size])
        for o in range(0, dataset_size, sub_size)
    ]


class DataLoader:
    """データをモデルに渡す処理をするクラス。

    継承してカスタマイズする。ここでData Augmentationとかをする。
    このクラス自体はステートレスにしておき、Iteratorが状態を持つ。

    Args:
        batch_size: バッチサイズ
        data_per_sample: sampleあたりのデータ数。mixupとかするなら2にする。
        parallel: サンプル単位でスレッド並列するならTrue

    """

    def __init__(self, batch_size: int = 16, data_per_sample=1, parallel: bool = True):
        self.batch_size = batch_size
        self.data_per_sample = data_per_sample
        self.parallel = parallel

    def iter(
        self, dataset: Dataset, shuffle: bool = False, use_horovod: bool = False
    ) -> Iterator:
        """Iteratorを作成する。

        Args:
            dataset: データセット
            shuffle: シャッフルするのか否か
            use_horovod: 1エポックあたりのミニバッチ数(__len__の戻り値)の算出にHorovodを考慮するか否か。

        Returns:
            Iterator

        """
        if shuffle:
            return RandomIterator(
                self, dataset, self.batch_size, self.data_per_sample, use_horovod
            )
        assert self.data_per_sample == 1  # 挙動がややこしいので1のみ可とする
        return SequentialIterator(self, dataset, self.batch_size, use_horovod)

    def get_batch(self, dataset: Dataset, indices: typing.Sequence[int]):
        """1件のミニバッチを取得する。

        Args:
            dataset: データセット
            indices: Iteratorが作成したindexの配列。

        Returns:
            ミニバッチ。通常は入力データとラベルのtuple。

        """
        # data
        if self.parallel:
            futures = [
                tk.threading.get_pool().submit(self.get_data, dataset, i)
                for i in indices
            ]
            data = [f.result() for f in futures]
        else:
            data = [self.get_data(dataset, i) for i in indices]
        # samples
        samples = self.collate_data(data)
        # batch
        return self.collate_samples(samples)

    def collate_data(self, data: list) -> list:
        """self.data_per_sample個ずつ集約してget_sampleする処理。"""
        return [
            self.get_sample(data[i : i + self.data_per_sample])
            for i in range(0, len(data), self.data_per_sample)
        ]

    def get_sample(self, data: list) -> tuple:
        """1件のサンプルを取得する。"""
        assert len(data) == self.data_per_sample
        assert self.data_per_sample == 1
        return data[0]

    def get_data(self, dataset: Dataset, index: int):
        """1件のデータを取得する。

        Args:
            dataset: データセット
            batch: Datasetが返したデータをバッチサイズ分集めたもの

        Returns:
            1件のデータ。通常は入力データとラベルのtuple。

        """
        return dataset.get_data(index)

    def collate_samples(self, batch: list) -> tuple:
        """バッチサイズ分のデータを集約する処理。

        Args:
            batch: get_sample()の結果をバッチサイズ分集めたもの

        Returns:
            モデルに渡されるデータ。通常は入力データとラベルのtuple。

        """
        return tuple(self.collate_part(part) for part in zip(*batch))

    def collate_part(self, part):
        """サンプルごとのデータをバッチ分まとめる処理。"""
        if isinstance(part[0], list):
            part = [np.array([x[i] for x in part]) for i in range(len(part[0]))]
        elif isinstance(part[0], dict):
            part = {k: np.array([x[k] for x in part]) for k in part[0]}
        else:
            part = np.array(part)
        return part


class Iterator(tf.keras.utils.Sequence, metaclass=abc.ABCMeta):
    """データをモデルに渡すクラス。

    Args:
        data_loader: データをモデルに渡す処理をするクラス
        dataset: データセット
        batch_size: バッチサイズ
        use_horovod: 1エポックあたりのミニバッチ数(__len__の戻り値)の算出にHorovodを考慮するか否か。

    Attributes:
        seconds_per_step: 1ステップ当たりの実処理時間の指数移動平均値

    """

    def __init__(
        self,
        data_loader: DataLoader,
        dataset: Dataset,
        batch_size: int,
        use_horovod: bool,
    ):
        self.data_loader = data_loader
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_horovod = use_horovod
        self.epoch: int = 1
        self.seconds_per_step: float = 0.0

    def on_epoch_end(self) -> None:
        """1エポック完了時の処理。"""
        self.epoch += 1

    def __len__(self) -> int:
        """1エポックあたりのミニバッチ数を返す。"""
        bs = self.batch_size * tk.hvd.size() if self.use_horovod else self.batch_size
        return -(-len(self.dataset) // bs)

    def __getitem__(self, index: int) -> tk.models.ModelIOType:
        """ミニバッチを1つ返す。"""
        start_time = time.perf_counter()

        indices = self.sample_batch_indices(index)
        batch = self.data_loader.get_batch(self.dataset, indices)

        elapsed_time = time.perf_counter() - start_time
        self.seconds_per_step = self.seconds_per_step * 0.99 + elapsed_time * 0.01
        return batch

    @abc.abstractmethod
    def sample_batch_indices(self, index: int) -> typing.Sequence[int]:
        """1ミニバッチ分のindexの配列を返す。"""

    def __iter__(self):
        """データを返す。"""
        for i in range(len(self)):
            yield self[i]

    def run(self):
        """データを返す。"""
        while True:
            for i in range(len(self)):
                yield self[i]
            self.on_epoch_end()


class SequentialIterator(Iterator):
    """順番にサンプリングするIterator。"""

    def sample_batch_indices(self, index: int) -> typing.Sequence[int]:
        """1ミニバッチ分のindexの配列を返す。"""
        start = self.batch_size * index
        end = start + self.batch_size
        return list(range(start, min(end, len(self.dataset))))


class RandomIterator(Iterator):
    """シャッフルするIterator。"""

    def __init__(
        self,
        data_loader: DataLoader,
        dataset: Dataset,
        batch_size: int,
        data_per_sample: int,
        use_horovod: bool,
    ):
        super().__init__(
            data_loader=data_loader,
            dataset=dataset,
            batch_size=batch_size,
            use_horovod=use_horovod,
        )
        self.data_per_sample = data_per_sample
        self.gens = [
            RandomIterator._generate_shuffled_indices(len(dataset))
            for _ in range(self.data_per_sample)
        ]
        self.indices = [
            [next(gen) for _ in range(len(self) * self.batch_size)] for gen in self.gens
        ]

    def on_epoch_end(self) -> None:
        """1エポック完了時の処理。"""
        super().on_epoch_end()
        self.indices = [
            [next(gen) for _ in range(len(self) * self.batch_size)] for gen in self.gens
        ]

    def sample_batch_indices(self, index: int) -> typing.Sequence[int]:
        """1ミニバッチ分のindexの配列を返す。"""
        offset = self.batch_size * index
        return sum(
            [indices[offset : offset + self.batch_size] for indices in self.indices], []
        )

    @staticmethod
    def _generate_shuffled_indices(data_count):
        """シャッフルしたインデックスのgenerator"""
        indices = np.arange(data_count)
        while True:
            random.shuffle(indices)
            yield from indices


class WeightedRandomIterator(Iterator):
    """重み付き乱数によるSampler。"""

    def sample_batch_indices(self, index: int) -> typing.Sequence[int]:
        """1ミニバッチ分のindexの配列を返す。"""
        del index
        assert self.dataset.weights is not None
        return random.choices(
            range(len(self.dataset)), weights=self.dataset.weights, k=self.batch_size
        )
