"""画像処理関連"""
import pathlib
import warnings

import numpy as np

from . import generator, ml, ndimage


class ImageDataGenerator(generator.Generator):
    """画像データのgenerator。

    Xは画像のファイルパスの配列またはndarray。
    ndarrayの場合は、(BGRではなく)RGB形式で、samples×rows×cols×channels。

    # 引数
    - grayscale: グレースケールで読み込むならTrue、RGBならFalse

    # 使用例
    ```
    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.ProcessOutput(tk.ml.to_categorical(num_classes), batch_axis=True))
    gen.add(tk.image.Resize((300, 300)))
    gen.add(tk.image.Mixup(probability=1, num_classes=num_classes))
    gen.add(tk.image.RandomPadding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.5))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize((300, 300)))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomColorAugmentors(probability=0.5))
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.ProcessInput(tk.image.preprocess_input_abs1))
    ```

    """

    def __init__(self, grayscale=False, profile=False):
        super().__init__(profile=profile)
        self.add(LoadImage(grayscale))


def preprocess_input_mean(x: np.ndarray):
    """RGBそれぞれ平均値(定数)を引き算。

    `keras.applications.imagenet_utils.preprocess_input` のようなもの。(ただし `channels_last` 限定)
    `keras.applications`のVGG16/VGG19/ResNet50で使われる。
    """
    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
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


class LoadImage(generator.Operator):
    """画像のリサイズ。

    # 引数

    - grayscale: グレースケールで読み込むならTrue、RGBならFalse
    """

    def __init__(self, grayscale):
        self.grayscale = grayscale

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if isinstance(rgb, np.ndarray):
            # ndarrayならそのまま画像扱い
            rgb = np.copy(rgb).astype(np.float32)
        else:
            # ファイルパスなら読み込み
            assert isinstance(rgb, (str, pathlib.Path))
            rgb = ndimage.load(rgb, self.grayscale)
        assert len(rgb.shape) == 3
        assert rgb.shape[-1] == (1 if self.grayscale else 3)
        return rgb, y, w


class Resize(generator.Operator):
    """画像のリサイズ。

    # 引数

    image_size: (height, width)のタプル
    """

    def __init__(self, image_size, padding=None):
        self.image_size = image_size
        self.padding = padding
        assert len(self.image_size) == 2

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        rgb = ndimage.resize(rgb, self.image_size[1], self.image_size[0], padding=self.padding)
        return rgb, y, w


class ProcessInput(generator.Operator):
    """画像に対する任意の処理。

    # 引数

    func: 画像のndarrayを受け取り、処理結果を返す関数
    batch_axis: Trueの場合、funcに渡されるndarrayのshapeが(1, height, width, channels)になる。Falseなら(height, width, channels)。

    # 例1
    ```py
    gen.add(ProcessInput(tk.image.preprocess_input_abs1))
    ```

    # 例2
    ```py
    gen.add(ProcessInput(tk.image.preprocess_input_mean))
    ```

    # 例3
    ```py
    gen.add(ProcessInput(keras.applications.vgg16.preprocess_input, batch_axis=True))
    ```
    """

    def __init__(self, func, batch_axis=False):
        self.func = func
        self.batch_axis = batch_axis

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if self.batch_axis:
            rgb = np.expand_dims(rgb, axis=0)
            rgb = self.func(rgb)
            rgb = np.squeeze(rgb, axis=0)
        else:
            rgb = self.func(rgb)
        return rgb, y, w


class ProcessOutput(generator.Operator):
    """ラベルに対する任意の処理。"""

    def __init__(self, func, batch_axis=False):
        self.func = func
        self.batch_axis = batch_axis

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if y is not None:
            if self.batch_axis:
                y = np.expand_dims(y, axis=0)
                y = self.func(y)
                y = np.squeeze(y, axis=0)
            else:
                y = self.func(y)
        return rgb, y, w


class RandomPadding(generator.Operator):
    """パディング。

    この後のRandomCropを前提に、パディングするサイズは固定。
    パディングのやり方がランダム。
    """

    def __init__(self, probability=1, padding_rate=0.25):
        assert 0 < probability <= 1
        self.probability = probability
        self.padding_rate = padding_rate

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            padding = rand.choice(('edge', 'zero', 'one', 'rand'))
            padded_w = int(np.ceil(rgb.shape[1] * (1 + self.padding_rate)))
            padded_h = int(np.ceil(rgb.shape[0] * (1 + self.padding_rate)))
            rgb = ndimage.pad(rgb, padded_w, padded_h, padding, rand)
        return rgb, y, w


class RandomRotate(generator.Operator):
    """回転。"""

    def __init__(self, probability=1, degrees=15):
        assert 0 < probability <= 1
        self.probability = probability
        self.degrees = degrees

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.rotate(rgb, rand.uniform(-self.degrees, self.degrees))
        return rgb, y, w


class RandomCrop(generator.Operator):
    """切り抜き。

    # Padding+Cropの例
    padding_rate=0.25、crop_rate=0.2で32px四方の画像を処理すると、
    上下左右に4pxずつパディングした後に、32px四方を切り抜く処理になる。
    256px四方だと、32pxパディングで256px四方を切り抜く。
    """

    def __init__(self, probability=1, crop_rate=0.4, aspect_prob=0.5, aspect_rations=(3 / 4, 4 / 3)):
        assert 0 < probability <= 1
        self.probability = probability
        self.crop_rate = crop_rate
        self.aspect_prob = aspect_prob
        self.aspect_rations = aspect_rations

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            cr = rand.uniform(1 - self.crop_rate, 1)
            ar = np.sqrt(rand.choice(self.aspect_rations)) if rand.rand() <= self.aspect_prob else 1
            cropped_w = min(int(np.floor(rgb.shape[1] * cr * ar)), rgb.shape[1])
            cropped_h = min(int(np.floor(rgb.shape[0] * cr / ar)), rgb.shape[0])
            crop_x = rand.randint(0, rgb.shape[1] - cropped_w + 1)
            crop_y = rand.randint(0, rgb.shape[0] - cropped_h + 1)
            rgb = ndimage.crop(rgb, crop_x, crop_y, cropped_w, cropped_h)
        return rgb, y, w


class RandomAugmentors(generator.Operator):
    """順番と適用確率をランダムにDataAugmentationを行う。

    # 引数
    augmentors: Augmentorの配列
    clip_rgb: RGB値をnp.clip(rgb, 0, 255)するならTrue
    """

    def __init__(self, augmentors, probability=1, clip_rgb=True):
        assert 0 < probability <= 1
        self.probability = probability
        self.augmentors = augmentors
        self.clip_rgb = clip_rgb

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            augmentors = self.augmentors[:]
            rand.shuffle(augmentors)
            for a in augmentors:
                rgb, y, w = a.execute(rgb, y, w, rand, ctx)
                assert rgb.dtype == np.float32, f'dtype error: {a.__class__}'
            # 色が範囲外になっていたら補正(飽和)
            if self.clip_rgb:
                rgb = np.clip(rgb, 0, 255)
        return rgb, y, w


class RandomColorAugmentors(RandomAugmentors):
    """色関連のDataAugmentationをいくつかまとめたもの。"""

    def __init__(self, probability=1):
        argumentors = [
            RandomBlur(probability=probability),
            RandomUnsharpMask(probability=probability),
            GaussianNoise(probability=probability),
            RandomSaturation(probability=probability),
            RandomBrightness(probability=probability),
            RandomContrast(probability=probability),
            RandomHue(probability=probability),
        ]
        super().__init__(argumentors, probability=1, clip_rgb=True)


class RandomFlipLR(generator.Operator):
    """左右反転。"""

    def __init__(self, probability=1):
        assert 0 < probability <= 1
        self.probability = probability

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.flip_lr(rgb)
            if y is not None and isinstance(y, ml.ObjectsAnnotation):
                y.bboxes[:, [0, 2]] = 1 - y.bboxes[:, [2, 0]]
        return rgb, y, w


class RandomFlipLRTB(generator.Operator):
    """左右反転 or 上下反転。"""

    def __init__(self, probability=1):
        assert 0 < probability <= 1
        self.probability = probability

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            if rand.rand() < 0.5:
                rgb = ndimage.flip_lr(rgb)
                if y is not None and isinstance(y, ml.ObjectsAnnotation):
                    y.bboxes[:, [0, 2]] = 1 - y.bboxes[:, [2, 0]]
            else:
                rgb = ndimage.flip_tb(rgb)
                if y is not None and isinstance(y, ml.ObjectsAnnotation):
                    y.bboxes[:, [1, 3]] = 1 - y.bboxes[:, [3, 1]]
        return rgb, y, w


class RandomBlur(generator.Operator):
    """ぼかし。"""

    def __init__(self, probability=1, radius=0.75):
        assert 0 < probability <= 1
        self.probability = probability
        self.radius = radius

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.blur(rgb, self.radius * rand.rand())
        return rgb, y, w


class RandomUnsharpMask(generator.Operator):
    """シャープ化。"""

    def __init__(self, probability=1, sigma=0.5, min_alpha=1, max_alpha=2):
        assert 0 < probability <= 1
        self.probability = probability
        self.sigma = sigma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.unsharp_mask(rgb, self.sigma, rand.uniform(self.min_alpha, self.max_alpha))
        return rgb, y, w


class RandomMedian(generator.Operator):
    """メディアンフィルタ。"""

    def __init__(self, probability=1, sizes=(3,)):
        assert 0 < probability <= 1
        self.probability = probability
        self.sizes = sizes

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.median(rgb, rand.choice(self.sizes))
        return rgb, y, w


class GaussianNoise(generator.Operator):
    """ガウシアンノイズ。"""

    def __init__(self, probability=1, scale=5):
        assert 0 < probability <= 1
        self.probability = probability
        self.scale = scale

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.gaussian_noise(rgb, rand, self.scale)
        return rgb, y, w


class RandomBrightness(generator.Operator):
    """明度の変更。"""

    def __init__(self, probability=1, shift=32):
        assert 0 < probability <= 1
        self.probability = probability
        self.shift = shift

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.brightness(rgb, rand.uniform(-self.shift, self.shift))
        return rgb, y, w


class RandomContrast(generator.Operator):
    """コントラストの変更。"""

    def __init__(self, probability=1, var=0.25):
        assert 0 < probability <= 1
        self.probability = probability
        self.var = var

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.contrast(rgb, rand.uniform(1 - self.var, 1 + self.var))
        return rgb, y, w


class RandomSaturation(generator.Operator):
    """彩度の変更。"""

    def __init__(self, probability=1, var=0.5):
        assert 0 < probability <= 1
        self.probability = probability
        self.var = var

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb = ndimage.saturation(rgb, rand.uniform(1 - self.var, 1 + self.var))
        return rgb, y, w


class RandomHue(generator.Operator):
    """色相の変更。"""

    def __init__(self, probability=1, var=1 / 16, shift=8):
        assert 0 < probability <= 1
        self.probability = probability
        self.var = var
        self.shift = shift

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            alpha = rand.uniform(1 - self.var, 1 + self.var, (3,))
            beta = rand.uniform(- self.shift, + self.shift, (3,))
            rgb = ndimage.hue_lite(rgb, alpha, beta)
        return rgb, y, w


class RandomErasing(generator.Operator):
    """Random Erasing。

    https://arxiv.org/abs/1708.04896

    # 引数
    - object_aware: yがObjectsAnnotationのとき、各オブジェクト内でRandom Erasing。(論文によるとTrueとFalseの両方をやるのが良い)
    - object_aware_prob: 各オブジェクト毎のRandom Erasing率。全体の確率は1.0にしてこちらで制御する。

    """

    def __init__(self, probability=1, scale_low=0.02, scale_high=0.4, rate_1=1 / 3, rate_2=3, object_aware=False, object_aware_prob=0.5, max_tries=30):
        assert 0 < probability <= 1
        assert scale_low <= scale_high
        assert rate_1 <= rate_2
        self.probability = probability
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.rate_1 = rate_1
        self.rate_2 = rate_2
        self.object_aware = object_aware
        self.object_aware_prob = object_aware_prob
        self.max_tries = max_tries

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            bboxes = np.round(y.bboxes * np.array(rgb.shape)[[1, 0, 1, 0]]) if isinstance(y, ml.ObjectsAnnotation) else None
            if self.object_aware:
                assert bboxes is not None
                # bboxes同士の重なり判定
                inter = ml.is_intersection(bboxes, bboxes)
                inter[range(len(bboxes)), range(len(bboxes))] = False  # 自分同士は重なってないことにする
                # 各box内でrandom erasing。
                for i, b in enumerate(bboxes):
                    if (b[2:] - b[:2] <= 1).any():
                        warnings.warn(f'bboxサイズが不正: {y.filename}, {b}')
                        continue  # 安全装置：サイズが無いboxはskip
                    if rand.rand() <= self.object_aware_prob:
                        b = np.copy(b).astype(int)
                        # box内に含まれる他のboxを考慮
                        inter_boxes = np.copy(bboxes[inter[i]])
                        inter_boxes -= np.expand_dims(np.tile(b[:2], 2), axis=0)  # bに合わせて平行移動
                        # random erasing
                        rgb[b[1]:b[3], b[0]:b[2], :] = self._erase_random(rgb[b[1]:b[3], b[0]:b[2], :], rand, inter_boxes)
            else:
                # 画像全体でrandom erasing。
                rgb = self._erase_random(rgb, rand, bboxes)
        return rgb, y, w

    def _erase_random(self, rgb, rand, bboxes):
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

            rgb[y:y + h, x:x + w, :] = rand.randint(0, 255, size=3)[np.newaxis, np.newaxis, :]
            break

        return rgb


class Mixup(generator.Operator):
    """`mixup`

    yはone-hot化済みの前提

    - mixup: Beyond Empirical Risk Minimization
      https://arxiv.org/abs/1710.09412

    """

    def __init__(self, probability=1, num_classes=None, alpha=0.2, beta=0.2):
        assert 0 < probability <= 1
        self.probability = probability
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            assert len(rgb.shape) == 3
            assert y is not None and len(y.shape) == 1
            # 混ぜる先を選ぶ
            ti = rand.randint(0, ctx.data_count)
            rgb2 = ndimage.resize(ndimage.load(ctx.X[ti]), rgb.shape[1], rgb.shape[0])
            y2 = ctx.y[ti]
            if self.num_classes is not None:
                t = np.zeros((self.num_classes,), dtype=y.dtype)
                t[y2] = 1
                y2 = t
            assert rgb.shape == rgb2.shape
            assert y.shape == y2.shape
            # 混ぜる
            m = rand.beta(self.alpha, self.beta)
            assert 0 <= m <= 1
            rgb = rgb * m + rgb2 * (1 - m)
            y = y * m + y2 * (1 - m)
        return rgb, y, w


class SamplewiseStandardize(generator.Operator):
    """標準化。0～255に適当に収める。"""

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        rgb = ndimage.standardize(rgb)
        return rgb, y, w


class ToGrayScale(generator.Operator):
    """グレースケール化。チャンネル数はとりあえず維持。"""

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        assert len(rgb.shape) == 3
        start_shape = rgb.shape
        rgb = ndimage.to_grayscale(rgb)
        rgb = np.tile(np.expand_dims(rgb, axis=-1), (1, 1, start_shape[-1]))
        assert rgb.shape == start_shape
        return rgb, y, w


class RandomBinarize(generator.Operator):
    """ランダム2値化(白黒化)。"""

    def __init__(self, threshold_min=128 - 32, threshold_max=128 + 32):
        assert 0 < threshold_min < 255
        assert 0 < threshold_max < 255
        assert threshold_min < threshold_max
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.data_augmentation:
            threshold = rand.uniform(self.threshold_min, self.threshold_max)
            rgb = ndimage.binarize(rgb, threshold)
        else:
            rgb = ndimage.binarize(rgb, (self.threshold_min + self.threshold_max) / 2)
        return rgb, y, w


class RotationsLearning(generator.Operator):
    """画像を0,90,180,270度回転させた画像を与え、その回転を推定する学習。

    Unsupervised Representation Learning by Predicting Image Rotations
    https://arxiv.org/abs/1803.07728

    # 使い方

    - `y` は `np.zeros((len(X),))` とする。
    - 4クラス分類として学習する。

    ```
    gen.add(tk.image.RotationsLearning())
    ```

    """

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        assert y == 0
        y = rand.randint(0, 4)
        rgb = ndimage.rot90(rgb, y)
        return rgb, y, w


class CustomOperator(generator.Operator):
    """カスタム処理用。"""

    def __init__(self, process):
        self.process = process

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        rgb, y, w = self.process(rgb, y, w, rand, ctx)
        return rgb, y, w


class CustomAugmentation(generator.Operator):
    """カスタム処理用。"""

    def __init__(self, process, probability=1):
        assert 0 < probability <= 1
        self.process = process
        self.probability = probability

    def execute(self, rgb, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            rgb, y, w = self.process(rgb, y, w, rand, ctx)
        return rgb, y, w
