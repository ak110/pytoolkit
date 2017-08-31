"""画像処理関連"""
import abc
import collections
import pathlib

import joblib  # pip install joblib
import numpy as np
import sklearn.utils

from . import dl, ndimage

# Augmentorと確率
AugmentorEntry = collections.namedtuple('AugmentorEntry', 'probability,augmentor')


class Augmentor(metaclass=abc.ABCMeta):
    """DataAugmentationを行うクラス。

    # 引数
    - partial: Trueにした場合、画像内のランダムな矩形を対象に処理を行う

    """

    def __init__(self, partial=False):
        self.partial = partial

    def execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        """DataAugmentationの実行。"""
        if self.partial:
            x1, x2 = self._get_partial_1d(rand, 0, rgb.shape[1])
            y1, y2 = self._get_partial_1d(rand, 0, rgb.shape[0])
            rgb[x1:x2, y1:y2, :] = self._execute(rgb[x1:x2, y1:y2, :], rand)
        else:
            rgb = self._execute(rgb, rand)
        assert rgb.dtype == np.float32
        return rgb

    @staticmethod
    def _get_partial_1d(rand, min_value: int, max_value: int):
        """`min_value`以上`max_value`未満の区間(幅1以上)をランダムに作って返す。"""
        assert min_value + 1 < max_value
        while True:
            v1 = rand.randint(min_value, max_value)
            v2 = rand.randint(min_value, max_value)
            if abs(v1 - v2) <= 1:
                continue
            return (v1, v2) if v1 < v2 else (v2, v1)

    @abc.abstractmethod
    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        """DataAugmentationの実装。"""
        pass


class ImageDataGenerator(dl.Generator):
    """画像データのgenerator。

    Xは画像のファイルパスの配列またはndarray。
    ndarrayの場合は、(BGRではなく)RGB形式で、samples×rows×cols×channels。

    # 引数
    - image_size: 出力する画像のサイズのタプル(rows, cols)
    - grayscale: グレースケールで読み込むならTrue、RGBならFalse
    - preprocess_input: 前処理を行う関数。Noneなら`keras.application.inception_v3.preprocess_input`風処理。(画素値を-1～+1に線形変換)

    # 使用例
    ```
    gen = tk.image.ImageDataGenerator((300, 300))
    gen.add(0.5, tk.image.FlipLR())
    gen.add(0.125, tk.image.RandomBlur())
    gen.add(0.125, tk.image.RandomBlur(partial=True))
    gen.add(0.125, tk.image.RandomUnsharpMask())
    gen.add(0.125, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.125, tk.image.RandomMedian())
    gen.add(0.125, tk.image.RandomMedian(partial=True))
    gen.add(0.125, tk.image.GaussianNoise())
    gen.add(0.125, tk.image.GaussianNoise(partial=True))
    gen.add(0.125, tk.image.RandomSaturation())
    gen.add(0.125, tk.image.RandomBrightness())
    gen.add(0.125, tk.image.RandomContrast())
    gen.add(0.125, tk.image.RandomLighting())
    ```

    """

    def __init__(self, image_size=(300, 300), grayscale=False, preprocess_input=None,
                 rotate_prob=0.125, rotate_degrees=15,
                 padding_rate=0.25, crop_rate=0.125,
                 aspect_rations=(1, 1, 3 / 4, 4 / 3)):
        self.image_size = image_size
        self.grayscale = grayscale
        self.preprocess_input = preprocess_input
        self.rotate_prob = rotate_prob
        self.rotate_degrees = rotate_degrees
        self.padding_rate = padding_rate
        self.crop_rate = crop_rate
        self.aspect_rations = aspect_rations
        self.augmentors = []

    def add(self, probability: float, augmentor: Augmentor):
        """Augmentorの追加"""
        self.augmentors.append(AugmentorEntry(probability=probability, augmentor=augmentor))

    def flow(self, X, y=None, weights=None, batch_size=32, shuffle=False, random_state=None, data_augmentation=False):  # pylint: disable=arguments-differ
        """`fit_generator`などに渡すgenerator。

        # 引数
        - data_augmentation: Data Augmentationを行うか否か。
        """
        random_state = sklearn.utils.check_random_state(random_state)
        random_state2 = np.random.RandomState(random_state.randint(0, 2 ** 31))
        with joblib.Parallel(n_jobs=batch_size, backend='threading') as parallel:
            for tpl in super().flow(X, y, weights, batch_size, shuffle, random_state,
                                    parallel=parallel, data_augmentation=data_augmentation, rand=random_state2):
                yield tpl

    def _prepare(self, X, y=None, weights=None, parallel=None, data_augmentation=False, rand=None):  # pylint: disable=arguments-differ
        """画像の読み込みとDataAugmentation。"""
        seeds = rand.randint(0, 2 ** 31, len(X))
        jobs = [joblib.delayed(self._load)(x, data_augmentation, seed) for x, seed in zip(X, seeds)]
        X = np.array(parallel(jobs))
        return super()._prepare(X, y, weights)

    def _load(self, x, data_augmentation, seed):
        """画像の読み込みとDataAugmentation。(単数)"""
        rand = np.random.RandomState(seed)  # 再現性やスレッドセーフ性に自信がないのでここではこれを使う。

        # 画像の読み込み
        rgb = self._load_image(x)

        # Data Augmentation
        if data_augmentation:
            # 変形を伴うData Augmentation
            rgb = self._transform(rgb, rand)
            # リサイズ
            rgb = ndimage.resize(rgb, self.image_size[1], self.image_size[0], padding=None)
            # 変形を伴わないData Augmentation
            augmentors = self.augmentors[:]
            rand.shuffle(augmentors)  # シャッフルして色々な順で適用
            for a in augmentors:
                if rand.rand() <= a.probability:
                    rgb = a.augmentor.execute(rgb, rand)
            # 色が範囲外になっていたら補正(飽和)
            rgb = np.clip(rgb, 0, 255)
        else:
            rgb = ndimage.resize(rgb, self.image_size[1], self.image_size[0], padding=None)

        # preprocess_input
        pi = self.preprocess_input if self.preprocess_input else preprocess_input_abs1
        rgb = np.expand_dims(rgb, axis=0)
        rgb = pi(rgb)
        rgb = np.squeeze(rgb, axis=0)
        return rgb

    def _load_image(self, x):
        """画像の読み込み"""
        if isinstance(x, np.ndarray):
            rgb = x.astype(np.float32)
            assert rgb.shape[-1] == (1 if self.grayscale else 3)
        else:
            assert isinstance(x, (str, pathlib.Path))
            color_mode = 'L' if self.grayscale else 'RGB'
            rgb = ndimage.load(x, color_mode)
        return rgb

    def _transform(self, rgb, rand):
        """変形を伴うAugmentation。"""
        # 回転
        if rand.rand() <= self.rotate_prob:
            rgb = ndimage.random_rotate(rgb, rand, degrees=self.rotate_degrees)
        # padding+crop
        rgb = ndimage.random_crop(rgb, rand, self.padding_rate, self.crop_rate, aspect_rations=self.aspect_rations)
        return rgb


def preprocess_input_mean(x: np.ndarray):
    """RGBそれぞれ平均値(定数)を引き算。

    `keras.applications.imagenet_utils.preprocess_input` のようなもの。(ただし `channels_last` 限定)
    `keras.applications`のVGG16/VGG19/ResNet50で使われる。
    """
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    return x


def preprocess_input_abs1(x: np.ndarray):
    """0～255を-1～1に変換。

    `keras.applications`のInceptionV3/Xceptionで使われる。
    """
    x /= 127.5
    x -= 1
    return x


class FlipLR(Augmentor):
    """左右反転。"""

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        assert rand is not None  # noqa
        return ndimage.flip_lr(rgb)


class FlipTB(Augmentor):
    """上下反転。"""

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        assert rand is not None  # noqa
        return ndimage.flip_tb(rgb)


class RandomBlur(Augmentor):
    """ぼかし。"""

    def __init__(self, radius=0.75, partial=False):
        self.radius = radius
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.blur(rgb, self.radius * rand.rand())


class RandomUnsharpMask(Augmentor):
    """シャープ化。"""

    def __init__(self, sigma=0.5, min_alpha=1, max_alpha=2, partial=False):
        self.sigma = sigma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.unsharp_mask(rgb, self.sigma, rand.uniform(self.min_alpha, self.max_alpha))


class RandomMedian(Augmentor):
    """メディアンフィルタ。"""

    def __init__(self, sizes=(2,), partial=False):
        self.sizes = sizes
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.median(rgb, rand.choice(self.sizes))


class GaussianNoise(Augmentor):
    """ガウシアンノイズ。"""

    def __init__(self, scale=5, partial=False):
        self.scale = scale
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.gaussian_noise(rgb, rand, self.scale)


class RandomSaturation(Augmentor):
    """彩度の変更。"""

    def __init__(self, var=0.25):
        self.var = var
        super().__init__()

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.saturation(rgb, rand.uniform(1 - self.var, 1 + self.var))


class RandomBrightness(Augmentor):
    """明度の変更。"""

    def __init__(self, var=0.25):
        self.var = var
        super().__init__()

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.brightness(rgb, rand.uniform(1 - self.var, 1 + self.var))


class RandomContrast(Augmentor):
    """コントラストの変更。"""

    def __init__(self, var=0.25):
        self.var = var
        super().__init__()

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.contrast(rgb, rand.uniform(1 - self.var, 1 + self.var))


class RandomLighting(Augmentor):
    """コントラストの変更。"""

    def __init__(self, std=0.5):
        self.std = std
        super().__init__()

    def _execute(self, rgb: np.ndarray, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.lighting(rgb, rand.randn(3) * self.std)
