"""学習時のデータの読み込み周り。"""
import abc
import concurrent.futures

import numpy as np

from . import keras

_THREAD_POOL = concurrent.futures.ThreadPoolExecutor()


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
            入力データとラベル。

            入力データが複数ある場合はlist形式で返す。
            出力が複数ある場合も同様にlist形式。
            単数の場合はnp.ndarrayである必要があるため注意。

            例えば、::

                input_a = keras.layers.Input((None,))
                input_b = keras.layers.Input((None,))
                x = ...
                x1 = keras.layers.Dense(1)(x)
                x2 = keras.layers.Dense(1)(x)
                model = keras.models.Model(inputs=[input_a, input_b], outputs=[x1, x2])

            上記のようなモデルにデータを渡す場合、::

                def __getitem__(self, index):
                    ...
                    return [x1, x2], [y1, y2]

            のように実装する。

        """
        raise NotImplementedError

    def __iter__(self):
        """データを返す。"""
        for i in range(len(self)):
            yield self[i]


class DataLoader(keras.utils.Sequence):
    """データをバッチサイズずつ読み込むクラス。ついでにオプションでmixupも。

    - mixup: Beyond Empirical Risk Minimization
      https://arxiv.org/abs/1710.09412

    Args:
        dataset (Dataset): データセット。
        batch_size (int): バッチサイズ。
        shuffle (bool): データをシャッフルするか否か。
        mp_size (int): プロセス数。
        parallel (bool): 並列処理を行うか否か。
        mixup (bool): mixupするか否か。

    """

    def __init__(self, dataset, batch_size, shuffle=False, mp_size=1, parallel=True, mixup=False):
        assert len(dataset) > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mp_size = mp_size
        self.parallel = parallel
        self.mixup = mixup
        bs = self.batch_size * self.mp_size
        self.steps_per_epoch = (len(self.dataset) + bs - 1) // bs
        if self.shuffle:
            self.index_generator = _generate_shuffled_indices(len(self.dataset))
            self.indices = [next(self.index_generator) for _ in range(self.steps_per_epoch * self.batch_size)]
        else:
            self.index_generator = None
            self.indices = np.arange(len(self.dataset))

    def on_epoch_end(self):
        """1エポック完了時の処理。(シャッフルする)"""
        if self.shuffle:
            self.indices = [next(self.index_generator) for _ in range(self.steps_per_epoch * self.batch_size)]

    def __len__(self):
        """1エポックあたりのミニバッチ数を返す。"""
        return self.steps_per_epoch

    def __getitem__(self, index):
        """指定されたインデックスのミニバッチを1件返す。

        Args:
            index (int): データのインデックス。

        Returns:
            入力データとラベル。

        """
        offset = self.batch_size * index
        batch_indices = self.indices[offset:offset + self.batch_size]
        assert not self.shuffle or len(batch_indices) == self.batch_size

        if self.parallel:
            futures = [_THREAD_POOL.submit(self.get_sample, i) for i in batch_indices]
            results = [f.result() for f in futures]
        else:
            results = [self.get_sample(i) for i in batch_indices]

        X_batch, y_batch = zip(*results)

        if isinstance(X_batch[0], list):
            X_batch = [np.array([x[i] for x in X_batch]) for i in range(len(X_batch[0]))]
        else:
            X_batch = np.array(X_batch)
        if isinstance(y_batch[0], list):
            y_batch = [np.array([y[i] for y in y_batch]) for i in range(len(y_batch[0]))]
        else:
            y_batch = np.array(y_batch)

        return X_batch, y_batch

    def get_sample(self, index):
        """指定されたインデックスのデータを返す。

        Args:
            index (int): データのインデックス。

        Returns:
            入力データとラベル。

        """
        X_i, y_i = self.dataset[index]

        if self.mixup:
            t = np.random.choice(len(self.dataset))
            X_t, y_t = self.dataset[t]
            r = np.float32(np.random.beta(0.2, 0.2))

            if isinstance(X_i, list):
                X_i = [X_i[i] * r + X_t[i] * (1 - r) for i in range(len(X_i))]
            else:
                X_i = (X_i * r + X_t * (1 - r))
            if isinstance(y_i, list):
                y_i = [y_i[i] * r + y_t[i] * (1 - r) for i in range(len(y_i))]
            else:
                y_i = (y_i * r + y_t * (1 - r))

        return X_i, y_i

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
