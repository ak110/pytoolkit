"""画像処理関連"""
import abc
import collections
import copy
import pathlib
import warnings

import numpy as np

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

    def execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        """DataAugmentationの実行。"""
        if self.partial:
            x1, x2 = self._get_partial_1d(rand, 0, rgb.shape[1])
            y1, y2 = self._get_partial_1d(rand, 0, rgb.shape[0])
            rgb[x1:x2, y1:y2, :] = self._execute(rgb[x1:x2, y1:y2, :], y, w, rand)
        else:
            rgb = self._execute(rgb, y, w, rand)
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
    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
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
    gen.add(0.5, tk.image.RandomErasing())
    gen.add(0.25, tk.image.RandomBlur())
    gen.add(0.25, tk.image.RandomBlur(partial=True))
    gen.add(0.25, tk.image.RandomUnsharpMask())
    gen.add(0.25, tk.image.RandomUnsharpMask(partial=True))
    gen.add(0.25, tk.image.RandomMedian())
    gen.add(0.25, tk.image.GaussianNoise())
    gen.add(0.25, tk.image.GaussianNoise(partial=True))
    gen.add(0.5, tk.image.RandomSaturation())
    gen.add(0.5, tk.image.RandomBrightness())
    gen.add(0.5, tk.image.RandomContrast())
    gen.add(0.5, tk.image.RandomHue())
    ```

    """

    def __init__(self, image_size=(300, 300), grayscale=False, preprocess_input=None,
                 rotate_prob=0.5, rotate_degrees=15,
                 padding_rate=0.25, crop_rate=0.15625,
                 aspect_prob=0.5,
                 aspect_rations=(3 / 4, 4 / 3),
                 data_encoder=None, label_encoder=None):
        self.image_size = image_size
        self.grayscale = grayscale
        self.preprocess_input = preprocess_input
        self.rotate_prob = rotate_prob
        self.rotate_degrees = rotate_degrees
        self.padding_rate = padding_rate
        self.crop_rate = crop_rate
        self.aspect_prob = aspect_prob
        self.aspect_rations = aspect_rations
        self.augmentors = []
        super().__init__(data_encoder, label_encoder)

    def add(self, probability: float, augmentor: Augmentor):
        """Augmentorの追加"""
        self.augmentors.append(AugmentorEntry(probability=probability, augmentor=augmentor))

    def generate(self, ix, seed, x_, y_, w_, data_augmentation):
        """画像の読み込みとDataAugmentation。(単数)"""
        rand = np.random.RandomState(seed)  # 再現性やスレッドセーフ性に自信がないのでここではこれを使う。
        y_ = copy.deepcopy(y_)
        w_ = copy.deepcopy(w_)

        # 画像の読み込み
        rgb, y_, w_ = self._load_image(x_, y_, w_)

        if data_augmentation:
            # 変形を伴うData Augmentation
            rgb, y_, w_ = self._transform(rgb, y_, w_, rand)
            # リサイズ
            interp = rand.choice(['nearest', 'lanczos', 'bilinear', 'bicubic'])
            rgb = ndimage.resize(rgb, self.image_size[1], self.image_size[0], padding=None, interp=interp)
        else:
            # リサイズ
            rgb = ndimage.resize(rgb, self.image_size[1], self.image_size[0], padding=None)

        # 変形を伴わないData Augmentation
        if data_augmentation:
            augmentors = self.augmentors[:]
            rand.shuffle(augmentors)  # シャッフルして色々な順で適用
            for a in augmentors:
                if rand.rand() <= a.probability:
                    rgb = a.augmentor.execute(rgb, y_, w_, rand)
            # 色が範囲外になっていたら補正(飽和)
            rgb = np.clip(rgb, 0, 255)

        # preprocess_input
        pi = self.preprocess_input if self.preprocess_input else preprocess_input_abs1
        rgb = np.expand_dims(rgb, axis=0)
        rgb = pi(rgb)
        rgb = np.squeeze(rgb, axis=0)
        return super().generate(ix, seed, rgb, y_, w_, data_augmentation)

    def _load_image(self, x, y, w):
        """画像の読み込み"""
        if isinstance(x, np.ndarray):
            rgb = x.astype(np.float32)
            assert rgb.shape[-1] == (1 if self.grayscale else 3)
        else:
            assert isinstance(x, (str, pathlib.Path))
            rgb = ndimage.load(x, self.grayscale)
        return rgb, y, w

    def _transform(self, rgb: np.ndarray, y, w, rand: np.random.RandomState):
        """変形を伴うAugmentation。"""
        # 回転
        if rand.rand() <= self.rotate_prob:
            padding = rand.choice(('same', 'zero', 'reflect', 'wrap'))
            rgb = ndimage.random_rotate(rgb, rand, degrees=self.rotate_degrees, padding=padding)
        # padding+crop
        padding = rand.choice(('same', 'zero', 'reflect', 'wrap', 'rand'))
        rgb = ndimage.random_crop(rgb, rand, self.padding_rate, self.crop_rate,
                                  self.aspect_prob, self.aspect_rations, padding=padding)
        return rgb, y, w


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


def unpreprocess_input_abs1(x: np.ndarray):
    """`preprocess_input_abs1`の逆変換。"""
    x += 1
    x *= 127.5
    return x


class FlipLR(Augmentor):
    """左右反転。"""

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        assert rand is not None  # noqa
        return ndimage.flip_lr(rgb)


class FlipTB(Augmentor):
    """上下反転。"""

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        assert rand is not None  # noqa
        return ndimage.flip_tb(rgb)


class RandomBlur(Augmentor):
    """ぼかし。"""

    def __init__(self, radius=0.75, partial=False):
        self.radius = radius
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.blur(rgb, self.radius * rand.rand())


class RandomUnsharpMask(Augmentor):
    """シャープ化。"""

    def __init__(self, sigma=0.5, min_alpha=1, max_alpha=2, partial=False):
        self.sigma = sigma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.unsharp_mask(rgb, self.sigma, rand.uniform(self.min_alpha, self.max_alpha))


class RandomMedian(Augmentor):
    """メディアンフィルタ。"""

    def __init__(self, sizes=(2,), partial=False):
        self.sizes = sizes
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.median(rgb, rand.choice(self.sizes))


class GaussianNoise(Augmentor):
    """ガウシアンノイズ。"""

    def __init__(self, scale=5, partial=False):
        self.scale = scale
        super().__init__(partial)

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.gaussian_noise(rgb, rand, self.scale)


class RandomBrightness(Augmentor):
    """明度の変更。"""

    def __init__(self, shift=32):
        self.shift = shift
        super().__init__()

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.brightness(rgb, rand.uniform(-self.shift, self.shift))


class RandomContrast(Augmentor):
    """コントラストの変更。"""

    def __init__(self, var=0.25):
        self.var = var
        super().__init__()

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.contrast(rgb, rand.uniform(1 - self.var, 1 + self.var))


class RandomSaturation(Augmentor):
    """彩度の変更。"""

    def __init__(self, var=0.5):
        self.var = var
        super().__init__()

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        return ndimage.saturation(rgb, rand.uniform(1 - self.var, 1 + self.var))


class RandomHue(Augmentor):
    """色相の変更。"""

    def __init__(self, var=0.25, shift=32):
        self.var = var
        self.shift = shift
        super().__init__()

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        alpha = rand.uniform(1 - self.var, 1 + self.var, (3,))
        beta = rand.uniform(- self.shift, + self.shift, (3,))
        return ndimage.hue_lite(rgb, alpha, beta)


class RandomErasing(Augmentor):
    """Random Erasing。

    https://arxiv.org/abs/1708.04896

    # 引数
    - object_aware: yがObjectsAnnotationのとき、各オブジェクト内でRandom Erasing。(論文によるとTrueとFalseの両方をやるのが良い)
    - object_aware_prob: 各オブジェクト毎のRandom Erasing率。全体の確率は1.0にしてこちらで制御する。

    """

    def __init__(self, scale_low=0.02, scale_high=0.4, rate_1=1 / 3, rate_2=3, object_aware=False, object_aware_prob=0.5, max_tries=30):
        assert scale_low <= scale_high
        assert rate_1 <= rate_2
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.rate_1 = rate_1
        self.rate_2 = rate_2
        self.object_aware = object_aware
        self.object_aware_prob = object_aware_prob
        self.max_tries = max_tries
        super().__init__()

    def _execute(self, rgb: np.ndarray, y, w, rand: np.random.RandomState) -> np.ndarray:
        from . import ml
        bboxes = np.round(y.bboxes * np.array(rgb.shape)[[1, 0, 1, 0]]) if isinstance(y, ml.ObjectsAnnotation) else None
        if self.object_aware:
            assert bboxes is not None
            # bboxes同士の重なり判定
            inter = ml.is_intersection(bboxes, bboxes)
            inter[range(len(bboxes)), range(len(bboxes))] = False  # 自分同士は重なってないことにする
            # 各box内でrandom erasing。
            for i, b in enumerate(bboxes):
                if (b[2:] - b[:2] <= 1).any():
                    warnings.warn('bboxサイズが不正: {}, {}'.format(y.filename, b))
                    continue  # 安全装置：サイズが無いboxはskip
                if rand.rand() <= self.object_aware_prob:
                    b = np.copy(b).astype(int)
                    # box内に含まれる他のboxを考慮
                    inter_boxes = np.copy(bboxes[inter[i]])
                    inter_boxes -= np.expand_dims(np.tile(b[:2], 2), axis=0)  # bに合わせて平行移動
                    # random erasing
                    rgb[b[1]:b[3], b[0]:b[2], :] = self._erase_random(rgb[b[1]:b[3], b[0]:b[2], :], rand, inter_boxes)

            return rgb
        else:
            # 画像全体でrandom erasing。
            return self._erase_random(rgb, rand, bboxes)

    def _erase_random(self, rgb, rand, bboxes):
        # from . import ml

        if bboxes is not None:
            bb_lt = bboxes[:, :2]  # 左上
            bb_rb = bboxes[:, 2:]  # 右下
            bb_lb = bboxes[:, (0, 3)]  # 左下
            bb_rt = bboxes[:, (1, 2)]  # 右上
            bb_c = (bb_lt + bb_rb) / 2  # 中央

        for _ in range(self.max_tries):
            s = rgb.shape[0] * rgb.shape[1] * rand.uniform(self.scale_low, self.scale_high)
            r = np.exp(rand.uniform(np.log(self.rate_1), np.log(self.rate_2)))
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            if w <= 0 or h <= 0 or w >= rgb.shape[1] or h >= rgb.shape[0]:
                continue
            x = rand.randint(0, rgb.shape[1] - w)
            y = rand.randint(0, rgb.shape[0] - h)

            if bboxes is not None:
                box_lt = np.array([[x, y]])
                box_rb = np.array([[x + w, y + h]])
                # bboxの頂点および中央を1つでも含んでいたらNGとする
                if np.logical_and(box_lt <= bb_lt, bb_lt <= box_rb).all(axis=-1).any() or \
                   np.logical_and(box_lt <= bb_rb, bb_rb <= box_rb).all(axis=-1).any() or \
                   np.logical_and(box_lt <= bb_lb, bb_lb <= box_rb).all(axis=-1).any() or \
                   np.logical_and(box_lt <= bb_rt, bb_rt <= box_rb).all(axis=-1).any() or \
                   np.logical_and(box_lt <= bb_c, bb_c <= box_rb).all(axis=-1).any():
                    continue
                # 面積チェック。塗りつぶされるのがbboxの面積の25%を超えていたらNGとする
                lt = np.maximum(bb_lt, box_lt)
                rb = np.minimum(bb_rb, box_rb)
                area_inter = np.prod(rb - lt, axis=-1) * (lt < rb).all(axis=-1)
                area_bb = np.prod(bb_rb - bb_lt, axis=-1)
                if (area_inter >= area_bb * 0.25).any():
                    continue

            rgb[y:y + h, x:x + w, :] = rand.randint(0, 255, size=(h, w, rgb.shape[2]))
            break

        return rgb
