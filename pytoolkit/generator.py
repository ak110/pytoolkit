"""データを読み込むgenerator。"""
import abc
import copy
import os
import time

import numpy as np
import sklearn.utils
import sklearn.externals.joblib as joblib


class GeneratorContext(object):
    """Generatorの中で使う情報をまとめて持ち歩くためのクラス。"""

    def __init__(self, X, y, weights, batch_size, shuffle, data_augmentation, random_state):
        self.X = X
        self.y = y
        self.weights = weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.data_count = len(X[0]) if isinstance(X, list) else len(X)
        if y is not None:
            assert (len(y[0]) if isinstance(y, list) else len(y)) == self.data_count
        if weights is not None:
            assert len(weights) == self.data_count

    def do_augmentation(self, rand, probability=1):
        """DataAugmentationを確率的にやるときの判定。"""
        return self.data_augmentation and (probability >= 1 or rand.rand() <= probability)


class Operator(metaclass=abc.ABCMeta):
    """ImageDataGeneratorで行う操作の基底クラス。"""

    @abc.abstractmethod
    def execute(self, rgb, y, w, rand, ctx: GeneratorContext):
        """処理。"""
        assert False


class Generator(object):
    """`fit_generator`などに渡すgeneratorを作るためのベースクラス。"""

    def __init__(self, profile=False):
        self.profile = profile
        self.profile_data = {}
        self.operators = []

    def add(self, operator: Operator):
        """Operatorの追加。"""
        self.operators.append(operator)

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, data_augmentation=False, random_state=None):
        """`fit_generator`などに渡すgenerator。kargsはそのままprepareに渡される。"""
        cpu_count = os.cpu_count()
        worker_count = min(batch_size, cpu_count * 3)
        ctx = GeneratorContext(X, y, weights, batch_size, shuffle, data_augmentation, random_state)
        with joblib.Parallel(n_jobs=worker_count, backend='threading') as parallel:
            for indices, seeds in self._flow_batch(ctx.data_count, ctx.batch_size, ctx.shuffle, ctx.random_state):
                batch = [joblib.delayed(self._work, check_pickle=False)(ix, seed, ctx) for ix, seed in zip(indices, seeds)]
                rx, ry, rw = zip(*parallel(batch))
                yield _get_result(ctx.X, ctx.y, ctx.weights, rx, ry, rw)

    def _flow_batch(self, data_count, batch_size, shuffle, random_state):
        """データのindexとseedをバッチサイズずつ列挙し続けるgenerator。"""
        if shuffle:
            # シャッフルありの場合、常にbatch_size分を返す (horovodとかはそれが都合が良い)
            batch_indices = []
            seeds = []
            for ix, seed in _flow_instance(data_count, shuffle, random_state):
                batch_indices.append(ix)
                seeds.append(seed)
                if len(batch_indices) == batch_size:
                    yield batch_indices, seeds
                    batch_indices = []
                    seeds = []
        else:
            # シャッフル無しの場合、1epoch分でぴったり終わるようにする (predict_generatorとか用)
            steps = self.steps_per_epoch(data_count, batch_size)
            indices = np.arange(data_count)
            while True:
                seeds = random_state.randint(0, 2 ** 31, size=(len(indices),))
                for batch_indices in np.array_split(indices, steps):
                    yield batch_indices, seeds[batch_indices]

    def _work(self, ix, seed, ctx: GeneratorContext):
        """1件1件の処理。"""
        x_, y_, w_ = _pick_next(ix, ctx.X, ctx.y, ctx.weights)
        rand = np.random.RandomState(seed)
        result_x, result_y, result_w = self._transform(x_, y_, w_, rand, ctx)
        assert result_x is not None
        assert (result_y is None) == (y_ is None)
        assert (result_w is None) == (w_ is None)
        return result_x, result_y, result_w

    def transform(self, x_, y_=None, w_=None, rand=None, data_augmentation=None):
        """1件分の処理を外から使うとき用のインターフェース。"""
        ctx = GeneratorContext(
            X=np.empty((0,)), y=None, weights=None, batch_size=None, shuffle=None,
            data_augmentation=data_augmentation, random_state=None)
        if rand is None:
            rand = ctx.random_state
        return self._transform(x_, y_, w_, rand, ctx)

    def _transform(self, x_, y_, w_, rand, ctx: GeneratorContext):
        """1件分の処理。

        画像の読み込みとかDataAugmentationとか。
        y_やw_は使わない場合もそのまま返せばOK。(使う場合はNoneに注意。)
        """
        x_ = copy.deepcopy(x_)  # 念のため
        y_ = copy.deepcopy(y_)  # 念のため
        w_ = copy.deepcopy(w_)  # 念のため
        if self.profile:
            for op in self.operators:
                start_time = time.time()
                x_, y_, w_ = op.execute(x_, y_, w_, rand, ctx)
                elapsed_time = time.time() - start_time
                key = op.__class__.__name__
                self.profile_data[key] = self.profile_data.get(key, 0) + elapsed_time
        else:
            for op in self.operators:
                x_, y_, w_ = op.execute(x_, y_, w_, rand, ctx)
        return x_, y_, w_

    @staticmethod
    def steps_per_epoch(data_count, batch_size):
        """1epochが何ステップかを算出して返す"""
        return (data_count + batch_size - 1) // batch_size

    def summary_profile(self, print_fn=print):
        """プロファイル結果のサマリ表示。"""
        assert self.profile
        total = sum(self.profile_data.values())
        for key, val in self.profile_data.items():
            print_fn('{:32s}: {:6.2f}%'.format(key, val * 100 / total))


def _flow_instance(data_count, shuffle, random_state):
    """データのindexとseedを1件ずつ列挙し続けるgenerator。"""
    indices = np.arange(data_count)
    while True:
        if shuffle:
            random_state.shuffle(indices)
        seeds = random_state.randint(0, 2 ** 31, size=(len(indices),))
        yield from zip(indices, seeds)


def _pick_next(ix, X, y, weights):
    """X, y, weightsからix番目の値を取り出す。"""
    def _pick(arr, ix):
        if arr is None:
            return None
        if isinstance(arr, list):
            return [x[ix] for x in arr]
        return arr[ix]

    return _pick(X, ix), _pick(y, ix), _pick(weights, ix)


def _get_result(X, y, weights, rx, ry, rw):
    """Kerasに渡すデータを返す。"""
    def _arr(arr, islist):
        if islist:
            return [np.array(a) for a in arr]
        return np.array(arr)

    if y is None:
        assert weights is None
        return _arr(rx, isinstance(X, list))
    elif weights is None:
        return _arr(rx, isinstance(X, list)), _arr(ry, isinstance(y, list))
    else:
        return _arr(rx, isinstance(X, list)), _arr(ry, isinstance(y, list)), np.array(rw)
