"""データを読み込むgenerator。"""
import abc
import concurrent.futures
import copy
import os
import time

import numpy as np
import sklearn.utils


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
        worker_count = min(batch_size * 4, cpu_count * 2)  # 適当に余裕をもったサイズにしておく
        ctx = GeneratorContext(X, y, weights, batch_size, shuffle, data_augmentation, random_state)
        with concurrent.futures.ThreadPoolExecutor(worker_count) as pool:
            yield from self._flow(pool, ctx)

    def _flow(self, pool, ctx: GeneratorContext):
        _MAX_QUEUE_BATCHES = 4  # ため込むバッチ数

        future_queue = []
        gen = self._flow_batch(ctx.data_count, ctx.batch_size, ctx.shuffle, ctx.random_state)
        try:
            while True:
                # 最大キューサイズまで仕事をsubmitする
                while len(future_queue) < _MAX_QUEUE_BATCHES:
                    indices, seeds = next(gen)
                    batch = [pool.submit(self._work, ix, seed, ctx) for ix, seed in zip(indices, seeds)]
                    future_queue.append(batch)

                # 先頭のバッチの処理結果を取り出す
                batch = [f.result() for f in future_queue[0]]
                rx, ry, rw = zip(*batch)
                yield _get_result(ctx.X, ctx.y, ctx.weights, rx, ry, rw)
                future_queue = future_queue[1:]
        except GeneratorExit:
            pass
        finally:
            gen.close()

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
        result_x, result_y, result_w = self.generate(seed, x_, y_, w_, ctx)
        assert result_x is not None
        assert (result_y is None) == (y_ is None)
        assert (result_w is None) == (w_ is None)
        return result_x, result_y, result_w

    def generate(self, seed, x_, y_, w_, ctx: GeneratorContext):
        """1件分の処理。

        画像の読み込みとかDataAugmentationとか。
        y_やw_は使わない場合もそのまま返せばOK。(使う場合はNoneに注意。)
        """
        rand = np.random.RandomState(seed)
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
        for ix, seed in zip(indices, seeds):
            yield ix, seed


def _pick_next(ix, X, y, weights):
    """X, y, weightsからix番目の値を取り出す。"""
    def _pick(arr, ix):
        if arr is None:
            return None
        return [x[ix] for x in arr] if isinstance(arr, list) else arr[ix]

    return _pick(X, ix), _pick(y, ix), _pick(weights, ix)


def _get_result(X, y, weights, rx, ry, rw):
    """Kerasに渡すデータを返す。"""
    def _arr(arr, islist):
        return [np.array(a) for a in arr] if islist else np.array(arr)

    if y is None:
        assert weights is None
        return _arr(rx, isinstance(X, list))
    elif weights is None:
        return _arr(rx, isinstance(X, list)), _arr(ry, isinstance(y, list))
    else:
        return _arr(rx, isinstance(X, list)), _arr(ry, isinstance(y, list)), np.array(rw)
