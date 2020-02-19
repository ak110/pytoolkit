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

# Dataset.dataの型
DataType = typing.Union[typing.Sequence[typing.Any], pd.DataFrame, dict]
# Dataset.labelsの型
LabelsType = typing.Union[
    np.ndarray, typing.List[np.ndarray], typing.Dict[str, np.ndarray]
]


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

    get_dataをオーバーライドすることで逐次読み込みなども可能とする。
    (ただし、sliceとかで__init__が呼ばれるので注意。)

    """

    data: DataType
    labels: LabelsType = None
    groups: np.ndarray = None
    weights: np.ndarray = None
    ids: np.ndarray = None
    init_score: np.ndarray = None
    metadata: typing.Optional[dict] = None

    def __len__(self) -> int:
        """データ件数を返す。"""
        return len(self.data)

    def get_data(self, index: int) -> typing.Tuple[typing.Any, typing.Any]:
        """dataとlabelを返す。"""
        if self.labels is None:
            return self.data[index], None
        return self._get(self.data, index), self._get(self.labels, index)

    def _get(self, data, index: int):
        """指定indexのデータ/ラベルを返す。"""
        if isinstance(data, dict):
            # multiple input/output
            return {k: v[index] for k, v in data.items()}
        elif isinstance(data, list):
            # multiple input/output
            return [v[index] for v in data]
        else:
            assert len(data) == len(self)
            return data[index]

    def iter(
        self, folds: tk.validation.FoldsType
    ) -> typing.Generator[typing.Tuple[Dataset, Dataset], None, None]:
        """foldsに従って分割する。"""
        for train_indices, val_indices in folds:
            train_set = self.slice(train_indices)
            val_set = self.slice(val_indices)
            yield train_set, val_set

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
        parallel: self.get_dataの呼び出しを並列化するか否か。

    """

    def __init__(self, batch_size: int = 16, data_per_sample=1, parallel=True):
        self.batch_size = batch_size
        self.data_per_sample = data_per_sample
        self.parallel = parallel

    def iter(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        without_label: bool = False,
        use_horovod: bool = False,
        num_replicas_in_sync: int = 1,
    ) -> Iterator:
        """Iteratorを作成する。

        Args:
            dataset: データセット
            shuffle: シャッフルするのか否か
            without_label: ラベルを使わない場合(predict)、Trueを指定する。
            use_horovod: 1エポックあたりのミニバッチ数(__len__の戻り値)の算出にHorovodを考慮するか否か。
            num_replicas_in_sync: tf.distribute使用時の並列数(バッチサイズに掛け算する)

        Returns:
            Iterator

        """
        assert len(dataset) >= 1
        ds = self.get_ds(dataset, shuffle, without_label, num_replicas_in_sync)
        bs = (
            self.batch_size
            * (tk.hvd.size() if use_horovod else 1)
            * num_replicas_in_sync
        )
        steps = -(-len(dataset) // bs)
        return Iterator(ds=ds, data_size=len(dataset), steps=steps)

    def get_ds(
        self,
        dataset: Dataset,
        shuffle: bool,
        without_label: bool,
        num_replicas_in_sync: int,
    ) -> tf.data.Dataset:
        """tf.data.Datasetを作る。"""
        # 試しに1件呼び出してdtypeやshapeを推定 (ダサいが…)
        exsample_data = self.get_data(dataset, 0)
        exsample_sample = self.get_sample(
            [exsample_data for _ in range(self.data_per_sample)]
        )
        assert (
            len(exsample_data) == 2
        ), f"get_data returns {len(exsample_data)} values, but expects to see 2 values. exsample_data={exsample_data}"
        assert (
            len(exsample_sample) == 2
        ), f"get_sample returns {len(exsample_sample)} values, but expects to see 2 values. exsample_data={exsample_sample}"
        data_tf_type = _get_tf_types(exsample_data)
        sample_tf_type = _get_tf_types(exsample_sample)

        def get_data(i):
            X, y = self.get_data(dataset, i)
            # tf.numpy_functionがNone未対応なので0にしちゃう
            if y is None:
                y = np.int32(0)
            # tf.numpy_functionがdict未対応なのでlistに展開してしまう
            # (並び順はexsample_dataに合わせる(一応))
            if isinstance(exsample_data[0], dict):
                X = [X[k] for k in exsample_data[0]]
            if isinstance(exsample_data[1], dict):
                y = [y[k] for k in exsample_data[1]]
            data = _flatten([X, y])
            return data

        def get_sample(*args):
            # flattenされたものをexsample_dataに従い戻す
            data_size = len(data_tf_type)
            assert len(args) % data_size == 0
            data_list = [
                _unflatten(exsample_data, args[i : i + data_size])
                for i in range(0, len(args), data_size)
            ]
            assert len(data_list) == self.data_per_sample, repr(data_list)

            X, y = self.get_sample(data_list)

            # tf.numpy_functionがNone未対応なので0にしちゃう
            if y is None:
                y = np.int32(0)
            # tf.numpy_functionがdict未対応なのでlistに展開してしまう
            # (並び順はexsample_dataに合わせる(一応))
            if isinstance(exsample_sample[0], dict):
                X = [X[k] for k in exsample_sample[0]]
            if isinstance(exsample_sample[1], dict):
                y = [y[k] for k in exsample_sample[1]]

            sample = _flatten([X, y])
            return sample

        def process1(i):
            data = tf.numpy_function(get_data, inp=[i], Tout=data_tf_type)
            return data

        def process2_1(*data):
            sample = tf.numpy_function(get_sample, inp=data, Tout=sample_tf_type)
            sample = _unflatten_tensor(exsample_sample, sample)
            if without_label:
                return sample[0]
            return sample

        def process2_2(data1, data2):
            sample = tf.numpy_function(
                get_sample, inp=(*data1, *data2), Tout=sample_tf_type
            )
            sample = _unflatten_tensor(exsample_sample, sample)
            if without_label:
                return sample[0]
            return sample

        ds = tf.data.Dataset.from_tensor_slices(np.arange(len(dataset)))
        num_parallel_calls = tf.data.experimental.AUTOTUNE if self.parallel else None
        if self.data_per_sample == 2:  # 挙動が複雑なので2のみ許可
            ds = tf.data.Dataset.zip(
                tuple(
                    (ds.shuffle(buffer_size=len(dataset)) if shuffle else ds).map(
                        process1, num_parallel_calls=num_parallel_calls,
                    )
                    for _ in range(self.data_per_sample)
                )
            )
            ds = ds.map(process2_2)
        else:
            assert self.data_per_sample == 1  # 挙動が複雑なので1のみ許可
            ds = ds.shuffle(buffer_size=len(dataset)) if shuffle else ds
            ds = ds.map(process1, num_parallel_calls=num_parallel_calls)
            ds = ds.map(process2_1)
        ds = ds.repeat() if shuffle else ds  # シャッフル時はバッチサイズを固定するため先にrepeat
        ds = ds.batch(self.batch_size * num_replicas_in_sync)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

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


def _flatten(a):
    """1次元配列化。"""
    if isinstance(a, (list, tuple)):
        return sum([_flatten(t) for t in a], [])
    assert not isinstance(a, dict)
    return [a]


def _get_tf_types(exsample_data):
    """exsample_dataからtf.dtypesの1次元リストを返す。"""
    if exsample_data is None:
        return [tf.int32]  # dummy
    elif isinstance(exsample_data, tuple):
        return sum([_get_tf_types(v) for v in exsample_data], [])
    elif isinstance(exsample_data, list):
        return sum([_get_tf_types(v) for v in exsample_data], [])
    elif isinstance(exsample_data, dict):
        # tf.numpy_functionがdict未対応なので、値の型だけリストで返す
        return sum([_get_tf_types(v) for v in exsample_data.values()], [])
    else:
        return [tf.dtypes.as_dtype(exsample_data.dtype)]


def _unflatten(exsample_data, data):
    """flattenされたdataをexsample_dataに従い戻す。"""
    if exsample_data is None:
        return data  # dummy
    elif isinstance(exsample_data, (tuple, list)):
        lengths = [
            len(v) if isinstance(v, (tuple, list, dict)) else 1 for v in exsample_data
        ]
        assert sum(lengths) == len(data), f"exsample_data={exsample_data} data={data}"
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        return tuple(
            _unflatten(v, data[o : o + l])
            for v, o, l in zip(exsample_data, offsets, lengths)
        )
    elif isinstance(exsample_data, dict):
        assert len(exsample_data) == len(
            data
        ), f"exsample_data={exsample_data} data={data}"
        return {k: _unflatten(v, d) for (k, v), d in zip(exsample_data.items(), data)}
    else:
        if isinstance(data, (tuple, list)):
            assert len(data) == 1, f"exsample_data={exsample_data} data={data}"
            data = data[0]
        return np.asarray(data, dtype=exsample_data.dtype)


def _unflatten_tensor(exsample_data, tensor):
    """numpyのサンプルデータに従いtf.ensure_shapeする。"""
    if exsample_data is None:
        return tf.ensure_shape(tensor, ())  # dummy
    elif isinstance(exsample_data, tuple):
        if len(exsample_data) == len(tensor):
            return tuple(_unflatten_tensor(v, t) for v, t in zip(exsample_data, tensor))
        else:
            # tf.numpy_functionがdict未対応なので展開しているのでここで戻す
            assert (
                len(exsample_data) == 2
            ), f"exsample_data={exsample_data} tensor={tensor}"
            if isinstance(exsample_data[0], (tuple, list, dict)):
                len1 = len(exsample_data[0])
                X = _unflatten_tensor(exsample_data[0], tensor[:len1])
            else:
                len1 = 1
                X = _unflatten_tensor(exsample_data[0], tensor[0])
            if isinstance(exsample_data[1], (tuple, list, dict)):
                len2 = len(exsample_data[1])
                y = _unflatten_tensor(exsample_data[1], tensor[len1:])
            else:
                len2 = 1
                y = _unflatten_tensor(exsample_data[1], tensor[len1])
            assert len1 + len2 == len(
                tensor
            ), f"exsample_data={exsample_data} tensor={tensor}"
            return X, y
    elif isinstance(exsample_data, list):
        assert len(exsample_data) == len(
            tensor
        ), f"exsample_data={exsample_data} tensor={tensor}"
        return [_unflatten_tensor(v, t) for v, t in zip(exsample_data, tensor)]
    elif isinstance(exsample_data, dict):
        # tf.numpy_functionがdict未対応なのでtensorはlistになっている
        assert len(exsample_data) == len(
            tensor
        ), f"exsample_data={exsample_data} tensor={tensor}"
        return {
            k: _unflatten_tensor(v, t)
            for (k, v), t in zip(exsample_data.items(), tensor)
        }
    else:
        return tf.ensure_shape(tensor, [None] * exsample_data.ndim)


@dataclasses.dataclass()
class Iterator:
    """データをモデルに渡すためのクラス。

    Args:
        ds: tf.data.Dataset
        data_size: データ数
        steps: 1エポックあたりのミニバッチ数

    """

    ds: tf.data.Dataset
    data_size: int
    steps: int

    def to_str(self) -> str:
        """情報を文字列化して返す。"""
        return f"element_spec={self.ds.element_spec} data_size={self.data_size} steps={self.steps}"


def mixup(
    ds: tf.data.Dataset,
    premix_fn: typing.Callable,
    postmix_fn: typing.Callable,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    data_count: int = None,
):
    """tf.dataでのmixup: <https://arxiv.org/abs/1710.09412>

    Args:
        ds: 元のデータセット
        premix_fn: DataAugmentationなどの処理
        postmix_fn: mixup後の処理
        num_parallel_calls: premix_fnの並列数
        data_count: シャッフル時のバッファサイズ

    """

    @tf.function
    def mixup_fn(data1, data2):
        r = tf.random.uniform((), 0, 1)
        return postmix_fn(
            *[
                tf.cast(d1, tf.float32) * r + tf.cast(d2, tf.float32) * (1 - r)
                for d1, d2 in zip(premix_fn(data1), premix_fn(data2))
            ]
        )

    data_count = data_count or tf.data.experimental.cardinality(ds)
    ds1 = ds.shuffle(buffer_size=data_count)
    ds2 = ds.shuffle(buffer_size=data_count)
    ds = tf.data.Dataset.zip((ds1, ds2))
    ds = ds.map(mixup_fn, num_parallel_calls=num_parallel_calls)
    return ds
