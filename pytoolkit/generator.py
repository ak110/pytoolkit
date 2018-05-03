"""データを読み込むgenerator。"""
import abc
import copy
import os
import time

import numpy as np
import sklearn.externals.joblib as joblib
import sklearn.utils


class GeneratorContext(object):
    """Generatorの中で使う情報をまとめて持ち歩くためのクラス。"""

    def __init__(self, X, y, weights, batch_size, shuffle, data_augmentation, random_state, balanced):
        self.X = X
        self.y = y
        self.weights = weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.balanced = balanced
        self.data_count = len(X[0]) if isinstance(X, list) else len(X)
        if y is not None:
            assert (len(y[0]) if isinstance(y, list) else len(y)) == self.data_count
        if weights is not None:
            assert len(weights) == self.data_count
        if balanced:
            assert shuffle, 'balancedはshuffle=Trueのときのみ有効'

    def do_augmentation(self, rand, probability=1):
        """DataAugmentationを確率的にやるときの判定。"""
        return self.data_augmentation and (probability >= 1 or rand.rand() <= probability)


class Operator(metaclass=abc.ABCMeta):
    """ImageDataGeneratorで行う操作の基底クラス。"""

    @abc.abstractmethod
    def execute(self, x, y, w, rand, ctx: GeneratorContext):
        """処理。"""
        return x, y, w


class Generator(object):
    """`fit_generator`などに渡すgeneratorを作るためのベースクラス。"""

    def __init__(self, profile=False):
        self.profile = profile
        self.profile_data = {}
        self.operators = []

    def add(self, operator: Operator):
        """Operatorの追加。"""
        self.operators.append(operator)

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, data_augmentation=False, random_state=None, balanced=False):
        """`fit_generator`などに渡すgenerator。

        # 引数
        - balanced: クラス間のバランスが均等になるようにサンプリングする。(shuffle=Trueな場合のみ有効。yはクラスのindexかそれをone-hot化したものなどである必要あり)

        """
        cpu_count = os.cpu_count()
        worker_count = min(batch_size, cpu_count * 3)
        ctx = GeneratorContext(X, y, weights, batch_size, shuffle, data_augmentation, random_state, balanced)
        with joblib.Parallel(n_jobs=worker_count, backend='threading') as parallel:
            for indices, seeds in self._flow_batch(ctx):
                batch = [joblib.delayed(self._work, check_pickle=False)(ix, seed, ctx) for ix, seed in zip(indices, seeds)]
                rx, ry, rw = zip(*parallel(batch))
                yield _get_result(ctx.X, ctx.y, ctx.weights, rx, ry, rw)

    def _flow_batch(self, ctx):
        """データのindexとseedをバッチサイズずつ列挙し続けるgenerator。"""
        if ctx.balanced:
            # クラス間のバランスが均等になるようにサンプリング
            assert ctx.shuffle
            assert isinstance(ctx.y, np.ndarray)
            assert len(ctx.y.shape) in (1, 2)
            classes = ctx.y if len(ctx.y.shape) == 1 else ctx.y.argmax(axis=-1)
            unique_classes = np.unique(classes)
            assert len(unique_classes) < len(classes)
            indices = np.arange(ctx.data_count)
            while True:
                y_batch = ctx.random_state.choice(classes, ctx.batch_size)
                batch_indices = [ctx.random_state.choice(indices[classes == y]) for y in y_batch]
                seeds = ctx.random_state.randint(0, 2 ** 31, size=(len(batch_indices),))
                yield batch_indices, seeds
        elif ctx.shuffle:
            # シャッフルありの場合、常にbatch_size分を返す (horovodとかはそれが都合が良い)
            batch_indices = []
            seeds = []
            for ix, seed in _flow_instance(ctx.data_count, ctx.shuffle, ctx.random_state):
                batch_indices.append(ix)
                seeds.append(seed)
                if len(batch_indices) == ctx.batch_size:
                    yield batch_indices, seeds
                    batch_indices = []
                    seeds = []
        else:
            # シャッフル無しの場合、1epoch分でぴったり終わるようにする (predict_generatorとか用)
            steps = self.steps_per_epoch(ctx.data_count, ctx.batch_size)
            indices = np.arange(ctx.data_count)
            while True:
                seeds = ctx.random_state.randint(0, 2 ** 31, size=(len(indices),))
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
            data_augmentation=data_augmentation, random_state=None, balanced=False)
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
            print_fn(f'{key:32s}: {val * 100 / total:6.2f}%')


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


class ProcessInput(Operator):
    """入力に対する任意の処理。

    # 引数

    func: 入力のndarrayを受け取り、処理結果を返す関数
    batch_axis: Trueの場合、funcに渡されるndarrayのshapeが(1, height, width, channels)になる。Falseなら(height, width, channels)。

    # 例1
    ```py
    gen.add(tk.generator.ProcessInput(tk.image.preprocess_input_abs1))
    ```

    # 例2
    ```py
    gen.add(tk.generator.ProcessInput(tk.image.preprocess_input_mean))
    ```

    # 例3
    ```py
    gen.add(tk.generator.ProcessInput(keras.applications.vgg16.preprocess_input, batch_axis=True))
    ```
    """

    def __init__(self, func, batch_axis=False):
        self.func = func
        self.batch_axis = batch_axis

    def execute(self, x, y, w, rand, ctx: GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if self.batch_axis:
            x = np.expand_dims(x, axis=0)
            x = self.func(x)
            x = np.squeeze(x, axis=0)
        else:
            x = self.func(x)
        return x, y, w


class ProcessOutput(Operator):
    """ラベルに対する任意の処理。"""

    def __init__(self, func, batch_axis=False):
        self.func = func
        self.batch_axis = batch_axis

    def execute(self, x, y, w, rand, ctx: GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if y is not None:
            if self.batch_axis:
                y = np.expand_dims(y, axis=0)
                y = self.func(y)
                y = np.squeeze(y, axis=0)
            else:
                y = self.func(y)
        return x, y, w


class CustomOperator(Operator):
    """カスタム処理用。"""

    def __init__(self, process):
        self.process = process

    def execute(self, x, y, w, rand, ctx: GeneratorContext):
        x, y, w = self.process(x, y, w, rand, ctx)
        return x, y, w


class CustomAugmentation(Operator):
    """カスタム処理用。"""

    def __init__(self, process, probability=1):
        assert 0 < probability <= 1
        self.process = process
        self.probability = probability

    def execute(self, x, y, w, rand, ctx: GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x, y, w = self.process(x, y, w, rand, ctx)
        return x, y, w


def mixup_generator(gen1, gen2, alpha=0.2, beta=0.2, random_state=None):
    """generator2つをmixupしたgeneratorを返す。

    - mixup: Beyond Empirical Risk Minimization
      https://arxiv.org/abs/1710.09412

    """
    random_state = sklearn.utils.check_random_state(random_state)
    for b1, b2 in zip(gen1, gen2):
        assert isinstance(b1, tuple)
        assert isinstance(b2, tuple)
        assert len(b1) in (2, 3)
        assert len(b2) in (2, 3)
        assert len(b1) == len(b2)
        # 混ぜる
        m = random_state.beta(alpha, beta)
        assert 0 <= m <= 1
        b = [x1 * m + x2 * (1 - m) for x1, x2 in zip(b1, b2)]
        yield b
