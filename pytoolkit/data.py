"""学習時のデータの読み込み周り。

dataset: dataの集合
data: 1件のデータ
sample: 学習時に使用する1件のデータ (1件以上のdataから作られるもの) (という事にしている)
batch: sampleのバッチサイズ個の集合

"""
from __future__ import annotations

import dataclasses
import random
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

    Args:
        batch_size: バッチサイズ
        data_per_sample: sampleあたりのデータ数。mixupとかするなら2にする。
        parallel: サンプル単位でスレッド並列するならTrue

    Attributes:
        get_data_Tout: self.get_dataの出力の型 (必要に応じて派生先で書き換える)
        get_sample_Tout: self.get_sampleの出力の型 (必要に応じて派生先で書き換える)

    """

    def __init__(self, batch_size: int = 16, data_per_sample=1, parallel: bool = True):
        self.batch_size = batch_size
        self.data_per_sample = data_per_sample
        self.parallel = parallel
        self.get_data_Tout = (tf.uint8, tf.float32)
        self.get_sample_Tout = (tf.float32, tf.float32)

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

        def get_data(i):
            return self.get_data(dataset, i)

        def get_sample(*args):
            return self.get_sample([args[i : i + 2] for i in range(0, len(args), 2)])

        data_size = len(dataset)
        ds = tf.data.Dataset.from_tensor_slices(np.arange(data_size))
        if shuffle and self.data_per_sample == 2:  # 挙動が複雑なので2のみ許可
            ds = tf.data.Dataset.zip(
                tuple(
                    ds.shuffle(buffer_size=data_size).map(
                        lambda i: tf.numpy_function(
                            get_data, inp=[i], Tout=self.get_data_Tout
                        ),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    )
                    for _ in range(self.data_per_sample)
                )
            )
            ds = ds.map(
                lambda data1, data2: tf.numpy_function(
                    get_sample, inp=(*data1, *data2), Tout=self.get_sample_Tout
                )
            )
        else:
            assert not shuffle or self.data_per_sample == 1  # 挙動が複雑なので1のみ許可
            ds = ds.shuffle(buffer_size=data_size) if shuffle else ds
            ds = ds.map(
                lambda i: tf.numpy_function(get_data, inp=[i], Tout=self.get_data_Tout),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            ds = ds.map(
                lambda *args: tf.numpy_function(
                    get_sample, inp=(*args,), Tout=self.get_sample_Tout
                )
            )
        ds = ds.repeat() if shuffle else ds  # シャッフル時はバッチサイズを固定するため先にrepeat
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        if use_horovod:
            samples_per_epoch = -(-data_size // (self.batch_size * tk.hvd.size()))
        else:
            samples_per_epoch = -(-data_size // self.batch_size)
        return Iterator(ds, data_size, self.batch_size, samples_per_epoch)

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


@dataclasses.dataclass()
class Iterator:
    """データをモデルに渡すためのクラス。

    Args:
        ds: tf.data.Dataset
        data_size: データ数
        batch_size: バッチサイズ
        steps_per_epoch: 1エポックあたりのミニバッチ数

    """

    ds: tf.data.Dataset
    data_size: int
    batch_size: int
    steps_per_epoch: int
