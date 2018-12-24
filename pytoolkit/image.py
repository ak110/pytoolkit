"""画像処理関連"""
import warnings

import numpy as np

from . import generator, math, ml, ndimage


class ImageDataGenerator(generator.Generator):
    """画像データのgenerator。

    Xは画像のファイルパスの配列またはndarray。
    ndarrayの場合は、(BGRではなく)RGB形式で、samples×rows×cols×channels。

    # 引数
    - grayscale: グレースケールで読み込むならTrue、RGBならFalse

    # 使用例
    ```
    gen = tk.image.ImageDataGenerator()
    gen.add(tk.image.Resize((300, 300)))
    gen.add(tk.image.Padding(probability=1))
    gen.add(tk.image.RandomRotate(probability=0.25))
    gen.add(tk.image.RandomCrop(probability=1))
    gen.add(tk.image.Resize((300, 300)))
    gen.add(tk.image.RandomFlipLR(probability=0.5))
    gen.add(tk.image.RandomColorAugmentors())
    gen.add(tk.image.RandomErasing(probability=0.5))
    gen.add(tk.image.Preprocess(mode='tf'))
    gen.add(tk.generator.ProcessOutput(tk.ml.to_categorical(num_classes), batch_axis=True))
    ```

    """

    def __init__(self, grayscale=False, use_cache=False, max_size=None, multiple_input=False, multiple_output=False, profile=False):
        super().__init__(multiple_input=multiple_input, multiple_output=multiple_output, profile=profile)
        self.add(LoadImage(grayscale=grayscale, use_cache=use_cache, max_size=max_size))


class LoadImage(generator.Operator):
    """画像の読み込み。

    # 引数

    - grayscale: グレースケールで読み込むならTrue、RGBならFalse
    - use_cache: 読み込み結果をdiskcacheライブラリでキャッシュするならTrue
    - max_size: このサイズを超えるなら縮小する。int or tuple。tupleは(height, width)
    """

    def __init__(self, grayscale=False, use_cache=False, max_size=None):
        self.grayscale = grayscale
        self.use_cache = use_cache
        self.max_size = max_size

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        x = ndimage.load(x, grayscale=self.grayscale, use_cache=self.use_cache, max_size=self.max_size)
        assert len(x.shape) == 3
        return x, y, w


class LoadOutputImage(generator.Operator):
    """画像の読み込み。

    # 引数

    - grayscale: グレースケールで読み込むならTrue、RGBならFalse
    - use_cache: 読み込み結果をdiskcacheライブラリでキャッシュするならTrue
    - max_size: このサイズを超えるなら縮小する。int or tuple。tupleは(height, width)
    """

    def __init__(self, grayscale=False, use_cache=False, max_size=None):
        self.grayscale = grayscale
        self.use_cache = use_cache
        self.max_size = max_size

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if y is not None:
            y = ndimage.load(y, grayscale=self.grayscale, use_cache=self.use_cache, max_size=self.max_size)
            assert len(y.shape) == 3
        return x, y, w


class Resize(generator.Operator):
    """画像のリサイズ。

    # 引数

    - image_size: (height, width)のタプル
    - padding: アスペクト比を保持するためにパディングするならパディングの種類

    """

    def __init__(self, image_size, padding=None, with_output=False):
        assert len(image_size) == 2
        self.image_size = image_size
        self.padding = padding
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        assert self.padding is None or not isinstance(y, ml.ObjectsAnnotation)  # paddingありで物体検出はとりあえず未対応
        x = ndimage.resize(x, self.image_size[1], self.image_size[0], padding=self.padding)
        if self.with_output and y is not None:
            y = ndimage.resize(y, self.image_size[1], self.image_size[0], padding=self.padding)
        return x, y, w


class Padding(generator.Operator):
    """パディング。

    この後のRandomCropなどを前提に、上下左右に固定サイズでパディング。
    """

    def __init__(self, mode='edge', probability=1, padding_rate=0.125, with_output=False):
        assert 0 < probability <= 1
        self.mode = mode
        self.probability = probability
        self.padding_rate = padding_rate
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        assert not isinstance(y, ml.ObjectsAnnotation)  # 物体検出は今のところ未対応
        if ctx.do_augmentation(rand, self.probability):
            padded_w = int(np.ceil(x.shape[1] * (1 + self.padding_rate)))
            padded_h = int(np.ceil(x.shape[0] * (1 + self.padding_rate)))
            x = ndimage.pad(x, padded_w, padded_h, self.mode)
            if self.with_output and y is not None:
                y = ndimage.pad(y, padded_w, padded_h, self.mode)
                assert x.shape[:2] == y.shape[:2]
        return x, y, w


class RandomPadding(generator.Operator):
    """ランダムなパディング。"""

    def __init__(self, mode='edge', probability=1, padding_rate=0.25, with_output=False):
        assert 0 < probability <= 1
        self.mode = mode
        self.probability = probability
        self.padding_rate = padding_rate
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        assert not isinstance(y, ml.ObjectsAnnotation)  # 物体検出は今のところ未対応
        if ctx.do_augmentation(rand, self.probability):
            pw = int(np.ceil(x.shape[1] * rand.rand() * self.padding_rate))
            ph = int(np.ceil(x.shape[0] * rand.rand() * self.padding_rate))
            px = rand.randint(0, pw + 1)
            py = rand.randint(0, ph + 1)
            x = ndimage.pad_ltrb(x, px, py, pw - px, ph - py, self.mode)
            if self.with_output and y is not None:
                y = ndimage.pad_ltrb(y, px, py, pw - px, ph - py, self.mode)
                assert x.shape[:2] == y.shape[:2]
        return x, y, w


class RandomRotate(generator.Operator):
    """回転。"""

    def __init__(self, probability=1, degrees=15, expand=True, border_mode='edge', with_output=False):
        assert 0 < probability <= 1
        self.probability = probability
        self.degrees = degrees
        self.expand = expand
        self.border_mode = border_mode
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        assert not isinstance(y, ml.ObjectsAnnotation)  # 物体検出は今のところ未対応
        if ctx.do_augmentation(rand, self.probability):
            degrees = rand.uniform(-self.degrees, self.degrees)
            x = ndimage.rotate(x, degrees, expand=self.expand, border_mode=self.border_mode)
            if self.with_output and y is not None:
                y = ndimage.rotate(y, degrees, expand=self.expand, border_mode=self.border_mode)
                assert x.shape[:2] == y.shape[:2]
        return x, y, w


class RandomCrop(generator.Operator):
    """切り抜き。

    # Padding+Cropの例
    padding_rate=0.125、crop_rate=0.2で32px四方の画像を処理すると、
    上下左右に2pxずつパディングした後に、32px四方を切り抜く処理になる。
    256px四方だと、32pxパディングで256px四方を切り抜く。
    """

    def __init__(self, probability=1, crop_rate=0.5, aspect_prob=0.5, aspect_rations=(3 / 4, 4 / 3), with_output=False):
        assert 0 < probability <= 1
        self.probability = probability
        self.crop_rate = crop_rate
        self.aspect_prob = aspect_prob
        self.aspect_rations = aspect_rations
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        assert not isinstance(y, ml.ObjectsAnnotation)  # 物体検出は今のところ未対応
        if ctx.do_augmentation(rand, self.probability):
            cr = rand.uniform(1 - self.crop_rate, 1)
            ar = rand.choice(self.aspect_rations) if rand.rand() <= self.aspect_prob else 1
            if ar <= 1:
                cropped_w = int(np.floor(x.shape[1] * cr * ar))
                cropped_h = int(np.floor(x.shape[0] * cr))
            else:
                cropped_w = int(np.floor(x.shape[1] * cr))
                cropped_h = int(np.floor(x.shape[0] * cr / ar))
            assert cropped_w <= x.shape[1] and cropped_h <= x.shape[0]
            crop_x = rand.randint(0, x.shape[1] - cropped_w + 1)
            crop_y = rand.randint(0, x.shape[0] - cropped_h + 1)
            x = ndimage.crop(x, crop_x, crop_y, cropped_w, cropped_h)
            if self.with_output and y is not None:
                y = ndimage.crop(y, crop_x, crop_y, cropped_w, cropped_h)
                assert x.shape[:2] == y.shape[:2]
        return x, y, w


class RandomZoom(generator.Operator):
    """Padding or crop + アスペクト比変更を行う。とりあえず物体検知用。

    # 引数

    - output_size: 出力画像サイズ (width, heightのタプル)
    - padding_rate: paddingする場合の面積の比の最大値。16なら最大で縦横4倍(SSD風)。
    - crop_rate: cropする場合の面積の比の最大値。0.1なら最小で縦横0.32倍(SSD風)。
    - keep_aspect: padding / cropの際にアスペクト比を保持するならTrue、正方形にリサイズしてしまうならFalse。
    - aspect_prob: アスペクト比を歪ませる確率。
    - max_aspect_ratio: アスペクト比を最大どこまで歪ませるか。(1.5なら正方形から3:2までランダムに歪ませる)
    - min_object_px: paddingなどでどこまでオブジェクトが小さくなるのを許容するか。(ピクセル数)

    padding_rateはNoneだとpaddingしない。
    crop_rateもNoneだとcropしない。

    """

    def __init__(self, probability=1, output_size=(300, 300),
                 padding_rate=16, crop_rate=0.1,
                 keep_aspect=False, aspect_prob=0.5, max_aspect_ratio=3 / 2,
                 min_object_px=8):
        assert 0 < probability <= 1
        assert max_aspect_ratio >= 1
        assert padding_rate is None or padding_rate > 1
        assert crop_rate is None or (0 < crop_rate < 1)
        self.probability = probability
        self.output_size = np.asarray(output_size)
        self.padding_rate = padding_rate
        self.crop_rate = crop_rate
        self.keep_aspect = keep_aspect
        self.aspect_prob = aspect_prob
        self.max_aspect_ratio = max_aspect_ratio
        self.min_aspect_ratio = 1 / self.max_aspect_ratio
        self.min_object_px = min_object_px

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            # padding or crop
            base_ar = x.shape[1] / x.shape[0] if self.keep_aspect else self.output_size[0] / self.output_size[1]
            rand_ar = np.exp(rand.uniform(np.log(self.min_aspect_ratio), np.log(self.max_aspect_ratio))) if rand.rand() <= self.aspect_prob else 1
            root_ar = np.sqrt(base_ar * rand_ar)
            ar = np.array([root_ar, 1 / root_ar])
            if self.padding_rate is not None and (self.crop_rate is None or rand.rand() <= 0.5):
                x = self._padding(x, y, rand, ar)
            elif self.crop_rate is not None:
                x = self._crop(x, y, rand, ar)

        # DataAugmentation無しの場合や失敗時のリサイズ
        if not all(x.shape[-2::-1] == self.output_size):
            if self.keep_aspect:
                # アスペクト比保持
                input_size = np.asarray(x.shape[-2::-1])
                resize_rate = (self.output_size / input_size).min()
                resized = np.floor(input_size * resize_rate + 0.1).astype(np.int32)
                assert all(resized <= self.output_size) and any(resized == self.output_size)
                x = ndimage.resize(x, resized[0], resized[1], padding=None)
                # パディング
                space = self.output_size - resized
                xy1 = space // 2
                xy2 = space - xy1
                assert all(xy1 + resized + xy2 == self.output_size)
                assert all(xy1) >= 0 and all(xy2 >= 0)
                x = ndimage.pad_ltrb(x, xy1[0], xy1[1], xy2[0], xy2[1], padding='mean')
                # 座標の修正
                if isinstance(y, ml.ObjectsAnnotation):
                    y.bboxes = (np.tile(xy1, 2) + y.bboxes * np.tile(resized, 2)) / np.tile(self.output_size, 2)
            else:
                # アスペクト比無視
                x = ndimage.resize(x, self.output_size[0], self.output_size[1], padding=None)

        assert all(x.shape[-2::-1] == self.output_size)
        return x, y, w

    def _padding(self, rgb, y, rand, ar):
        """Padding(zoom-out)。"""
        input_size = np.asarray(rgb.shape[-2::-1])
        for _ in range(30):
            pr = rand.uniform(1, np.sqrt(self.padding_rate))
            padded_size = np.ceil(input_size * pr * ar).astype(int)
            padded_size = np.maximum(padded_size, input_size)
            padding_size = padded_size - input_size
            paste_xy = np.array([rand.randint(0, padding_size[0] + 1), rand.randint(0, padding_size[1] + 1)])
            if isinstance(y, ml.ObjectsAnnotation):
                bboxes = np.copy(y.bboxes)
                bboxes = (np.tile(paste_xy, 2) + bboxes * np.tile(input_size, 2)) / np.tile(padded_size, 2)
                sb = bboxes * np.tile(self.output_size, 2)
                if (sb[:, 2:] - sb[:, :2] < self.min_object_px).any():  # あまりに小さいbboxが発生するのはNG
                    continue
                y.bboxes = bboxes
            # 先に縮小
            new_size = np.floor(input_size * self.output_size / padded_size).astype(int)
            rgb = ndimage.resize(rgb, new_size[0], new_size[1], padding=None)
            # パディング
            paste_lr = np.floor(paste_xy * self.output_size / padded_size).astype(int)
            paste_tb = self.output_size - (paste_lr + new_size)
            rgb = ndimage.pad_ltrb(rgb, paste_lr[0], paste_lr[1], paste_tb[0], paste_tb[1], 'mean')
            assert all(rgb.shape[-2::-1] == self.output_size)
            break
        return rgb

    def _crop(self, rgb, y, rand, ar):
        """Crop(zoom-in)。"""
        # SSDでは結構複雑なことをやっているが、とりあえず簡単に実装
        if isinstance(y, ml.ObjectsAnnotation):
            bb_center = ml.bboxes_center(y.bboxes)
            bb_area = ml.bboxes_area(y.bboxes)
        input_size = np.asarray(rgb.shape[-2::-1])
        for _ in range(30):
            cr = rand.uniform(np.sqrt(self.crop_rate), 1)  # SSD風：[0.1, 1]
            cropped_wh = np.floor(input_size * cr * ar).astype(int)
            cropped_wh = np.minimum(cropped_wh, input_size)
            cropping_size = input_size - cropped_wh
            crop_xy = np.array([rand.randint(0, cropping_size[0] + 1), rand.randint(0, cropping_size[1] + 1)])
            crop_box = np.concatenate([crop_xy, crop_xy + cropped_wh]) / np.tile(input_size, 2)
            if isinstance(y, ml.ObjectsAnnotation):
                # 中心を含むbboxのみ有効
                bb_mask = math.in_range(bb_center, crop_box[:2], crop_box[2:]).all(axis=-1)
                if not bb_mask.any():
                    continue
                # あまり極端に面積が減っていないbboxのみ有効
                lt = np.maximum(crop_box[np.newaxis, :2], y.bboxes[:, :2])
                rb = np.minimum(crop_box[np.newaxis, 2:], y.bboxes[:, 2:])
                cropped_area = (rb - lt).prod(axis=-1) * (lt < rb).all(axis=-1)
                bb_mask = np.logical_and(bb_mask, cropped_area >= bb_area * 0.3)
                bboxes = np.copy(y.bboxes)
                bboxes = (bboxes * np.tile(input_size, 2) - np.tile(crop_xy, 2)) / np.tile(cropped_wh, 2)
                bboxes = np.clip(bboxes, 0, 1)
                sb = bboxes * np.tile(self.output_size, 2)
                if (sb[:, 2:] - sb[:, :2] < self.min_object_px).any():  # あまりに小さいbboxが発生するのはNG
                    continue
                y.bboxes = bboxes[bb_mask]
                y.classes = y.classes[bb_mask]
                y.difficults = y.difficults[bb_mask]
            # 切り抜き
            rgb = ndimage.crop(rgb, crop_xy[0], crop_xy[1], cropped_wh[0], cropped_wh[1])
            assert (rgb.shape[-2::-1] == cropped_wh).all()
            # リサイズ
            rgb = ndimage.resize(rgb, self.output_size[0], self.output_size[1])
            assert all(rgb.shape[-2::-1] == self.output_size)
            break
        return rgb


class RandomAugmentors(generator.Operator):
    """順番と適用確率をランダムにDataAugmentationを行う。

    # 引数
    augmentors: Augmentorの配列
    """

    def __init__(self, augmentors, probability=1):
        assert 0 < probability <= 1
        self.probability = probability
        self.augmentors = augmentors

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            augmentors = self.augmentors[:]
            rand.shuffle(augmentors)
            for a in augmentors:
                x, y, w = a.execute(x, y, w, rand, ctx)
                assert x.dtype == np.uint8, f'dtype error: {a.__class__}'
        return x, y, w


class RandomColorAugmentors(RandomAugmentors):
    """色関連のDataAugmentationをいくつかまとめたもの。"""

    def __init__(self):
        argumentors = [
            RandomBlur(probability=0.125),
            RandomUnsharpMask(probability=0.125),
            GaussianNoise(probability=0.125),
            RandomSaturation(probability=0.25),
            RandomBrightness(probability=0.25),
            RandomContrast(probability=0.25),
            RandomHue(probability=0.25),
            RandomEqualize(probability=0.0625),
            RandomAutoContrast(probability=0.0625),
            RandomPosterize(probability=0.0625),
            RandomAlpha(probability=0.125),
        ]
        super().__init__(argumentors, probability=1)


class RandomFlipLR(generator.Operator):
    """左右反転。(`tk.ml.ObjectsAnnotation`対応)"""

    def __init__(self, probability=1, with_output=False):
        assert 0 < probability <= 1
        self.probability = probability
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.flip_lr(x)
            if self.with_output and y is not None:
                y = ndimage.flip_lr(y)
            elif y is not None and isinstance(y, ml.ObjectsAnnotation):
                y.bboxes[:, [0, 2]] = 1 - y.bboxes[:, [2, 0]]
        return x, y, w


class RandomFlipTB(generator.Operator):
    """上下反転。(`tk.ml.ObjectsAnnotation`対応)"""

    def __init__(self, probability=1, with_output=False):
        assert 0 < probability <= 1
        self.probability = probability
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.flip_tb(x)
            if self.with_output and y is not None:
                y = ndimage.flip_tb(y)
            if y is not None and isinstance(y, ml.ObjectsAnnotation):
                y.bboxes[:, [1, 3]] = 1 - y.bboxes[:, [3, 1]]
        return x, y, w


class RandomRotate90(generator.Operator):
    """90度/180度/270度回転。(`tk.ml.ObjectsAnnotation`対応)"""

    def __init__(self, probability=1, with_output=False):
        assert 0 < probability <= 1
        self.probability = probability
        self.with_output = with_output

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            k = rand.randint(0, 4)
            x = ndimage.rot90(x, k)
            if self.with_output and y is not None:
                y = ndimage.rot90(y, k)
            elif y is not None and isinstance(y, ml.ObjectsAnnotation):
                y.rot90(k)
        return x, y, w


class RandomBlur(generator.Operator):
    """ぼかし。"""

    def __init__(self, probability=1, radius=0.75):
        assert 0 < probability <= 1
        self.probability = probability
        self.radius = radius

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.blur(x, self.radius * rand.rand())
        return x, y, w


class RandomUnsharpMask(generator.Operator):
    """シャープ化。"""

    def __init__(self, probability=1, sigma=0.5, min_alpha=1, max_alpha=2):
        assert 0 < probability <= 1
        self.probability = probability
        self.sigma = sigma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.unsharp_mask(x, self.sigma, rand.uniform(self.min_alpha, self.max_alpha))
        return x, y, w


class RandomMedian(generator.Operator):
    """メディアンフィルタ。"""

    def __init__(self, probability=1, sizes=(3,)):
        assert 0 < probability <= 1
        self.probability = probability
        self.sizes = sizes

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.median(x, rand.choice(self.sizes))
        return x, y, w


class GaussianNoise(generator.Operator):
    """ガウシアンノイズ。"""

    def __init__(self, probability=1, scale=5):
        assert 0 < probability <= 1
        self.probability = probability
        self.scale = scale

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.gaussian_noise(x, rand, self.scale)
        return x, y, w


class RandomBrightness(generator.Operator):
    """明度の変更。"""

    def __init__(self, probability=1, shift=32):
        assert 0 < probability <= 1
        self.probability = probability
        self.shift = shift

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.brightness(x, rand.uniform(-self.shift, self.shift))
        return x, y, w


class RandomContrast(generator.Operator):
    """コントラストの変更。"""

    def __init__(self, probability=1, var=0.25):
        assert 0 < probability <= 1
        self.probability = probability
        self.var = var

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.contrast(x, rand.uniform(1 - self.var, 1 + self.var))
        return x, y, w


class RandomSaturation(generator.Operator):
    """彩度の変更。"""

    def __init__(self, probability=1, var=0.5):
        assert 0 < probability <= 1
        self.probability = probability
        self.var = var

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.saturation(x, rand.uniform(1 - self.var, 1 + self.var))
        return x, y, w


class RandomHue(generator.Operator):
    """色相の変更。"""

    def __init__(self, probability=1, var=1 / 16, shift=8):
        assert 0 < probability <= 1
        self.probability = probability
        self.var = var
        self.shift = shift

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            alpha = rand.uniform(1 - self.var, 1 + self.var, (3,))
            beta = rand.uniform(- self.shift, + self.shift, (3,))
            x = ndimage.hue_lite(x, alpha, beta)
        return x, y, w


class RandomEqualize(generator.Operator):
    """ヒストグラム平坦化。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def __init__(self, probability=1):
        assert 0 < probability <= 1
        self.probability = probability

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.equalize(x)
        return x, y, w


class RandomAutoContrast(generator.Operator):
    """オートコントラスト。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def __init__(self, probability=1):
        assert 0 < probability <= 1
        self.probability = probability

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            x = ndimage.auto_contrast(x)
        return x, y, w


class RandomPosterize(generator.Operator):
    """ポスタリゼーション。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def __init__(self, probability=1, min_bits=4, max_bits=7):
        assert 0 < probability <= 1
        self.probability = probability
        self.min_bits = min_bits
        self.max_bits = max_bits

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            bits = rand.randint(self.min_bits, self.max_bits + 1)
            x = ndimage.posterize(x, bits)
        return x, y, w


class Normalize(generator.Operator):
    """正規化。

    scale=2, shift=-0.5なら-1 ～ +1。
    scale=255, shift=-0.5なら-127.5 ～ +127.5。
    scale=1, shift=0なら0 ～ 1。
    """

    def __init__(self, scale=2, shift=-0.5):
        self.scale = scale
        self.shift = shift

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        x = ndimage.auto_contrast(x, self.scale) + self.scale * self.shift
        return x, y, w


class RandomAlpha(generator.Operator):
    """画像の一部にランダムな色の半透明の矩形を描画する。"""

    def __init__(self, probability=1, alpha=0.125, scale_low=0.02, scale_high=0.4, rate_1=1 / 3, rate_2=3, max_tries=30):
        assert 0 < probability <= 1
        assert scale_low <= scale_high
        assert rate_1 <= rate_2
        self.probability = probability
        self.alpha = alpha
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.rate_1 = rate_1
        self.rate_2 = rate_2
        self.max_tries = max_tries

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            for _ in range(self.max_tries):
                s = x.shape[0] * x.shape[1] * rand.uniform(self.scale_low, self.scale_high)
                r = np.exp(rand.uniform(np.log(self.rate_1), np.log(self.rate_2)))
                ew = int(np.sqrt(s / r))
                eh = int(np.sqrt(s * r))
                if ew <= 0 or eh <= 0 or ew >= x.shape[1] or eh >= x.shape[0]:
                    continue
                ex = rand.randint(0, x.shape[1] - ew)
                ey = rand.randint(0, x.shape[0] - eh)
                rc = rand.randint(0, 256, size=x.shape[-1])
                x_ref = x[ey:ey + eh, ex:ex + ew, :]
                x[ey:ey + eh, ex:ex + ew, :] = x_ref * (1 - self.alpha) + rc * self.alpha
                break
        return x, y, w


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

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            bboxes = np.round(y.bboxes * np.array(x.shape)[[1, 0, 1, 0]]) if isinstance(y, ml.ObjectsAnnotation) else None
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
                        x[b[1]:b[3], b[0]:b[2], :] = self._erase_random(x[b[1]:b[3], b[0]:b[2], :], rand, inter_boxes)
            else:
                # 画像全体でrandom erasing。
                x = self._erase_random(x, rand, bboxes)
        return x, y, w

    def _erase_random(self, x, rand, bboxes):
        if bboxes is not None:
            bb_lt = bboxes[:, :2]  # 左上
            bb_rb = bboxes[:, 2:]  # 右下
            bb_lb = bboxes[:, (0, 3)]  # 左下
            bb_rt = bboxes[:, (1, 2)]  # 右上
            bb_c = (bb_lt + bb_rb) / 2  # 中央

        for _ in range(self.max_tries):
            s = x.shape[0] * x.shape[1] * rand.uniform(self.scale_low, self.scale_high)
            r = np.exp(rand.uniform(np.log(self.rate_1), np.log(self.rate_2)))
            ew = int(np.sqrt(s / r))
            eh = int(np.sqrt(s * r))
            if ew <= 0 or eh <= 0 or ew >= x.shape[1] or eh >= x.shape[0]:
                continue
            ex = rand.randint(0, x.shape[1] - ew)
            ey = rand.randint(0, x.shape[0] - eh)

            if bboxes is not None:
                box_lt = np.array([[ex, ey]])
                box_rb = np.array([[ex + ew, ey + eh]])
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

            x[ey:ey + eh, ex:ex + ew, :] = rand.randint(0, 256, size=x.shape[-1])[np.newaxis, np.newaxis, :]
            break

        return x


class Mixup(generator.Operator):
    """`mixup`

    yはone-hot化済みの前提

    - mixup: Beyond Empirical Risk Minimization
      https://arxiv.org/abs/1710.09412

    # 引数
    - alpha: α
    - beta: β
    - data_loader: X[i]と返すべきshapeを受け取り、データを読み込んで返す。
    - num_classes: クラス数 (指定した場合、one-hot化を行う)

    """

    def __init__(self, probability=1, alpha=0.2, beta=0.2, data_loader=None, num_classes=None):
        assert 0 < probability <= 1
        self.probability = probability
        self.alpha = alpha
        self.beta = beta
        self.data_loader = data_loader or self._load_data
        self.num_classes = num_classes

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.do_augmentation(rand, self.probability):
            assert y is not None
            # 混ぜる先を選ぶ
            ti = rand.randint(0, ctx.data_count)
            x2 = self.data_loader(ctx.X[ti], x.shape)
            y2 = ctx.y[ti]
            if self.num_classes is not None:
                t = np.zeros((self.num_classes,), dtype=y.dtype)
                t[y2] = 1
                y2 = t
            assert x.shape == x2.shape
            assert y.shape == y2.shape
            # 混ぜる
            m = rand.beta(self.alpha, self.beta)
            assert 0 <= m <= 1
            x = x * m + x2 * (1 - m)
            y = y * m + y2 * (1 - m)
        return x, y, w

    def _load_data(self, x, shape):
        """画像の読み込み"""
        assert self is not None
        return ndimage.resize(ndimage.load(x), shape[1], shape[0])


class SamplewiseStandardize(generator.Operator):
    """標準化。0～255に適当に収める。"""

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        x = ndimage.standardize(x)
        return x, y, w


class ToGrayScale(generator.Operator):
    """グレースケール化。チャンネル数はとりあえず維持。"""

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        assert len(x.shape) == 3
        start_shape = x.shape
        x = ndimage.to_grayscale(x)
        x = np.tile(np.expand_dims(x, axis=-1), (1, 1, start_shape[-1]))
        assert x.shape == start_shape
        return x, y, w


class RandomBinarize(generator.Operator):
    """ランダム2値化(白黒化)。"""

    def __init__(self, threshold_min=128 - 32, threshold_max=128 + 32):
        assert 0 < threshold_min < 255
        assert 0 < threshold_max < 255
        assert threshold_min < threshold_max
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        if ctx.data_augmentation:
            threshold = rand.uniform(self.threshold_min, self.threshold_max)
            x = ndimage.binarize(x, threshold)
        else:
            x = ndimage.binarize(x, (self.threshold_min + self.threshold_max) / 2)
        return x, y, w


class RotationsLearning(generator.Operator):
    """画像を0,90,180,270度回転させた画像を与え、その回転を推定する学習。

    Unsupervised Representation Learning by Predicting Image Rotations
    https://arxiv.org/abs/1803.07728

    # 使い方

    - `y` は `np.zeros((len(X),))` とする。
    - 4クラス分類として学習する。
    - 一番最後に `gen.add(tk.image.RotationsLearning())`

    """

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        assert y == 0
        k = rand.randint(0, 4)
        x = ndimage.rot90(x, k)
        y = np.zeros((4,))
        y[k] = 1
        return x, y, w


class Preprocess(generator.Operator):
    """画像データの前処理。

    # 引数

    - mode: 'none', 'caffe', 'tf', 'torch', 'div255'
    """

    def __init__(self, mode='tf'):
        assert mode in ('none', 'caffe', 'tf', 'torch', 'div255')
        self.mode = mode

    def execute(self, x, y, w, rand, ctx: generator.GeneratorContext):
        """処理。"""
        assert rand is not None  # noqa
        if self.mode == 'none':
            pass
        elif self.mode == 'caffe':
            x = x[..., ::-1] + np.array([-103.939, -116.779, -123.68])
        elif self.mode == 'tf':
            x = (x / 127.5) - 1
        elif self.mode == 'torch':
            x = ((x / 255.) + np.array([-0.485, -0.456, -0.406])) / np.array([0.229, 0.224, 0.225])
        elif self.mode == 'div255':
            x = x / 255.
        return x, y, w
