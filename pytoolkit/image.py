"""画像処理関連。"""
# pylint: disable=arguments-differ,abstract-method
from __future__ import annotations

import random
import warnings

import albumentations as A
import numpy as np
import scipy.ndimage

import pytoolkit as tk


class RandomCompose(A.Compose):
    """シャッフル付きCompose。"""

    def __call__(self, force_apply=False, **data):
        """変換の適用。"""
        backup = self.transforms.transforms.copy()
        try:
            random.shuffle(self.transforms.transforms)
            return super().__call__(force_apply=force_apply, **data)
        finally:
            self.transforms.transforms = backup


class RandomRotate(A.DualTransform):
    """回転。"""

    def __init__(
        self, degrees=15, expand=True, border_mode="edge", always_apply=False, p=0.5
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.degrees = degrees
        self.expand = expand
        self.border_mode = border_mode

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "keypoints": self.apply_to_keypoints,
        }

    def apply(self, image, degrees, **params):
        return tk.ndimage.rotate(
            image, degrees, expand=self.expand, border_mode=self.border_mode
        )

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError()

    def apply_to_keypoint(self, keypoint, **params):
        # TODO
        raise NotImplementedError()

    def get_params(self):
        return {"degrees": random.uniform(-self.degrees, self.degrees)}

    def get_transform_init_args_names(self):
        return ("degrees", "expand", "border_mode")


class RandomTransform(A.DualTransform):
    """Flip, Scale, Resize, Rotateをまとめて処理。"""

    @classmethod
    def create_refine(
        cls,
        width,
        height,
        flip_h=True,
        flip_v=False,
        translate_h=0.0625,
        translate_v=0.0625,
        border_mode="edge",
        always_apply=False,
        clip_bboxes=True,
        p=1,
    ):
        """Refined Data Augmentation <https://arxiv.org/abs/1909.09148> 用の控えめバージョンを作成する。"""
        return cls(
            width=width,
            height=height,
            flip_h=flip_h,
            flip_v=flip_v,
            translate_h=translate_h,
            translate_v=translate_v,
            border_mode=border_mode,
            scale_prob=0,
            aspect_prob=0,
            rotate_prob=0,
            clip_bboxes=clip_bboxes,
            always_apply=always_apply,
            p=p,
        )

    def __init__(
        self,
        width,
        height,
        flip_h=True,
        flip_v=False,
        translate_h=0.125,
        translate_v=0.125,
        scale_prob=0.5,
        scale_range=(2 / 3, 3 / 2),
        base_scale=1.0,
        aspect_prob=0.5,
        aspect_range=(3 / 4, 4 / 3),
        rotate_prob=0.25,
        rotate_range=(-15, +15),
        interp="lanczos",
        border_mode="edge",
        clip_bboxes=True,
        always_apply=False,
        p=1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.width = width
        self.height = height
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.translate_h = translate_h
        self.translate_v = translate_v
        self.scale_prob = scale_prob
        self.base_scale = base_scale
        self.scale_range = scale_range
        self.aspect_prob = aspect_prob
        self.aspect_range = aspect_range
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.interp = interp
        self.border_mode = border_mode
        self.clip_bboxes = clip_bboxes

    def apply(self, image, m, interp=None, **params):
        return tk.ndimage.perspective_transform(
            image,
            self.width,
            self.height,
            m,
            interp=interp or self.interp,
            border_mode=self.border_mode,
        )

    def apply_to_bbox(self, bbox, m, image_size, **params):
        bbox = np.asarray(bbox)
        assert bbox.shape == (4,)
        bbox *= np.array([image_size[1], image_size[0]] * 2)
        bbox = tk.ndimage.transform_points(bbox, m)
        if bbox[2] < bbox[0]:
            bbox = bbox[[2, 1, 0, 3]]
        if bbox[3] < bbox[1]:
            bbox = bbox[[0, 3, 2, 1]]
        bbox /= np.array([self.width, self.height] * 2)
        assert bbox.shape == (4,)
        if self.clip_bboxes:
            bbox = np.clip(bbox, 0, 1)
        return tuple(bbox)

    def apply_to_keypoint(self, keypoint, m, **params):
        return tuple(tk.ndimage.transform_points(keypoint[:2], m)) + tuple(keypoint[2:])

    def apply_to_mask(self, img, interp=None, **params):
        del interp
        return self.apply(img, interp="nearest", **params)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        scale = (
            self.base_scale
            * np.exp(
                random.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1]))
            )
            if random.random() <= self.scale_prob
            else self.base_scale
        )
        ar = (
            np.exp(
                random.uniform(
                    np.log(self.aspect_range[0]), np.log(self.aspect_range[1])
                )
            )
            if random.random() <= self.aspect_prob
            else 1.0
        )
        m = tk.ndimage.compute_perspective(
            image.shape[1],
            image.shape[0],
            self.width,
            self.height,
            flip_h=self.flip_h and random.random() <= 0.5,
            flip_v=self.flip_v and random.random() <= 0.5,
            scale_h=scale * np.sqrt(ar),
            scale_v=scale / np.sqrt(ar),
            degrees=random.uniform(self.rotate_range[0], self.rotate_range[1])
            if random.random() <= self.rotate_prob
            else 0,
            pos_h=random.uniform(0, 1),
            pos_v=random.uniform(0, 1),
            translate_h=random.uniform(-self.translate_h, self.translate_h),
            translate_v=random.uniform(-self.translate_v, self.translate_v),
        )
        return {"m": m, "image_size": image.shape[:2]}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "width",
            "height",
            "flip_h",
            "flip_v",
            "translate_h",
            "translate_v",
            "scale_prob",
            "scale_range",
            "base_scale",
            "aspect_prob",
            "aspect_range",
            "rotate_prob",
            "rotate_range",
            "interp",
            "border_mode",
        )


class Resize(A.DualTransform):
    """リサイズ。"""

    def __init__(self, width, height, always_apply=False, p=1):
        super().__init__(always_apply=False, p=1)
        self.width = width
        self.height = height

    def apply(self, image, **params):
        return tk.ndimage.resize(image, width=self.width, height=self.height)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("width", "height")


class RandomColorAugmentors(RandomCompose):
    """色関連のDataAugmentationをいくつかまとめたもの。

    Args:
        noisy: Trueを指定すると細かいノイズ系も有効になる。
        grayscale: RGBではなくグレースケールならTrue。

    """

    def __init__(self, noisy: bool = False, grayscale: bool = False, p=1):
        argumentors = [
            RandomBrightness(p=0.25),
            RandomContrast(p=0.25),
            RandomHue(p=0.25),
            RandomSaturation(p=0.25),
            RandomEqualize(p=0.0625),
            RandomAutoContrast(p=0.0625),
            RandomAlpha(p=0.25),
        ]
        if noisy:
            argumentors.extend(
                [
                    RandomPosterize(p=0.0625),
                    A.Solarize(threshold=(50, 255 - 50), p=0.0625),
                    RandomBlur(p=0.125),
                    RandomUnsharpMask(p=0.125),
                    A.IAASharpen(alpha=(0, 0.5), p=0.125),
                    GaussNoise(p=0.125),
                    SpeckleNoise(p=0.125),
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.125),
                ]
            )
        if not grayscale and noisy:
            argumentors.extend(
                [A.ISONoise(color_shift=(0, 0.05), intensity=(0, 0.5), p=0.125)]
            )
        super().__init__(argumentors, p=p)

    def get_transform_init_args_names(self):
        return ("noisy", "grayscale")


class GaussNoise(A.ImageOnlyTransform):
    """ガウシアンノイズ。"""

    def __init__(self, scale=(0, 15), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        rand = np.random.RandomState(random.randrange(2 ** 32))
        return tk.ndimage.gaussian_noise(image, rand, scale)

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class SaltAndPepperNoise(A.ImageOnlyTransform):
    """Salt-and-pepper noise。"""

    def __init__(self, salt=0.01, pepper=0.01, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.salt = salt
        self.pepper = pepper

    def apply(self, image, **params):
        rand = np.random.RandomState(random.randrange(2 ** 32))
        map_ = rand.uniform(size=image.shape[:2])
        image = image.copy()
        image[map_ < self.salt] = 255
        image[map_ > 1 - self.pepper] = 0
        return image

    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ("salt", "pepper")


class PerlinNoise(A.ImageOnlyTransform):
    """Perlin noise。"""

    def __init__(
        self,
        alpha=(0, 0.5),
        frequency=(2.0, 4.0),
        octaves=(3, 5),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.frequency = frequency
        self.octaves = octaves

    def apply(self, image, frequency, octaves, seed, alpha, **params):
        noise = tk.ndimage.perlin_noise(
            image.shape[:2], frequency=frequency, octaves=octaves, seed=seed
        )
        if image.ndim != 2:
            assert image.ndim == 3
            noise = np.expand_dims(noise, axis=-1)
        image = (image * (1 - alpha) + noise * alpha).astype(image.dtype)
        return image

    def get_params(self):
        return {
            "frequency": random.uniform(*self.frequency),
            "octaves": random.randint(*self.octaves),
            "seed": random.randint(0, 2 ** 32 - 1),
            "alpha": random.uniform(*self.alpha),
        }

    def get_transform_init_args_names(self):
        return ("alpha", "frequency", "octaves")


class RandomBlur(A.ImageOnlyTransform):
    """ぼかし。"""

    def __init__(self, sigma=(0, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma

    def apply(self, image, sigma, **params):
        return tk.ndimage.blur(image, sigma)

    def get_params(self):
        return {"sigma": random.uniform(*self.sigma)}

    def get_transform_init_args_names(self):
        return ("sigma",)


class RandomUnsharpMask(A.ImageOnlyTransform):
    """シャープ化。"""

    def __init__(self, sigma=0.5, alpha=(0, 3), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma
        self.alpha = alpha

    def apply(self, image, alpha, **params):
        return tk.ndimage.unsharp_mask(image, self.sigma, alpha)

    def get_params(self):
        return {"alpha": random.uniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("sigma", "alpha")


class RandomBrightness(A.ImageOnlyTransform):
    """明度の変更。"""

    def __init__(self, shift=(-50, 50), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.shift = shift

    def apply(self, image, shift, **params):
        return tk.ndimage.brightness(image, shift)

    def get_params(self):
        return {"shift": random.uniform(*self.shift)}

    def get_transform_init_args_names(self):
        return ("shift",)


class RandomContrast(A.ImageOnlyTransform):
    """コントラストの変更。"""

    def __init__(self, alpha=(1 / 2, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, image, alpha, **params):
        return tk.ndimage.contrast(image, alpha)

    def get_params(self):
        return {"alpha": _random_loguniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)


class RandomSaturation(A.ImageOnlyTransform):
    """彩度の変更。"""

    def __init__(self, alpha=(1 / 2, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, image, alpha, **params):
        if image.shape[-1] != 3:
            return image
        return tk.ndimage.saturation(image, alpha)

    def get_params(self):
        return {"alpha": _random_loguniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)


class RandomHue(A.ImageOnlyTransform):
    """色相の変更。"""

    def __init__(self, alpha=(1 / 1.5, 1.5), beta=(-30, 30), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.beta = beta

    def apply(self, image, alpha, beta, **params):
        if image.shape[-1] != 3:
            return image
        return tk.ndimage.hue_lite(image, alpha, beta)

    def get_params(self):
        return {
            "alpha": np.array(
                [
                    _random_loguniform(*self.alpha),
                    _random_loguniform(*self.alpha),
                    _random_loguniform(*self.alpha),
                ]
            ),
            "beta": np.array(
                [
                    random.uniform(*self.beta),
                    random.uniform(*self.beta),
                    random.uniform(*self.beta),
                ]
            ),
        }

    def get_transform_init_args_names(self):
        return ("alpha", "beta")


class RandomEqualize(A.ImageOnlyTransform):
    """ヒストグラム平坦化。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, image, **params):
        return tk.ndimage.equalize(image)

    def get_transform_init_args_names(self):
        return ()


class RandomAutoContrast(A.ImageOnlyTransform):
    """オートコントラスト。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, image, **params):
        return tk.ndimage.auto_contrast(image)

    def get_transform_init_args_names(self):
        return ()


class RandomPosterize(A.ImageOnlyTransform):
    """ポスタリゼーション。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def __init__(self, bits=(4, 7), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.bits = bits

    def apply(self, image, bits, **params):
        return tk.ndimage.posterize(image, bits)

    def get_params(self):
        return {"bits": random.randint(*self.bits)}

    def get_transform_init_args_names(self):
        return ("bits",)


class RandomAlpha(A.ImageOnlyTransform):
    """画像の一部にランダムな色の半透明の矩形を描画する。"""

    def __init__(
        self,
        alpha=0.125,
        scale_low=0.02,
        scale_high=0.4,
        rate_1=1 / 3,
        rate_2=3,
        max_tries=30,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        assert scale_low <= scale_high
        assert rate_1 <= rate_2
        self.alpha = alpha
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.rate_1 = rate_1
        self.rate_2 = rate_2
        self.max_tries = max_tries

    def apply(self, image, **params):
        rand = np.random.RandomState(random.randrange(2 ** 32))
        return tk.ndimage.erase_random(
            image,
            rand,
            bboxes=None,
            scale_low=self.scale_low,
            scale_high=self.scale_high,
            rate_1=self.rate_1,
            rate_2=self.rate_2,
            alpha=self.alpha,
            max_tries=self.max_tries,
        )

    def get_transform_init_args_names(self):
        return ("alpha", "scale_low", "scale_high", "rate_1", "rate_2", "max_tries")


class RandomErasing(A.ImageOnlyTransform):
    """Random Erasing <https://arxiv.org/abs/1708.04896>

    Args:
        object_aware: yがObjectsAnnotationのとき、各オブジェクト内でRandom Erasing。(論文によるとTrueとFalseの両方をやるのが良い)
        object_aware_prob: 各オブジェクト毎のRandom Erasing率。全体の確率は1.0にしてこちらで制御する。

    """

    def __init__(
        self,
        scale_low=0.02,
        scale_high=0.4,
        rate_1=1 / 3,
        rate_2=3,
        object_aware=False,
        object_aware_prob=0.5,
        max_tries=30,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        assert scale_low <= scale_high
        assert rate_1 <= rate_2
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.rate_1 = rate_1
        self.rate_2 = rate_2
        self.object_aware = object_aware
        self.object_aware_prob = object_aware_prob
        self.max_tries = max_tries

    def apply(self, image, bboxes=None, **params):
        rand = np.random.RandomState(random.randrange(2 ** 32))
        image = np.copy(image)
        bboxes = (
            np.round(bboxes * np.array(image.shape)[[1, 0, 1, 0]])
            if bboxes is not None
            else None
        )
        if self.object_aware:
            assert bboxes is not None
            # bboxes同士の重なり判定
            inter = tk.od.is_intersection(bboxes, bboxes)
            inter[range(len(bboxes)), range(len(bboxes))] = False  # 自分同士は重なってないことにする
            # 各box内でrandom erasing。
            for i, b in enumerate(bboxes):
                if (b[2:] - b[:2] <= 1).any():
                    warnings.warn(f"bboxサイズが不正: {b}")
                    continue  # 安全装置：サイズが無いboxはskip
                if random.random() <= self.object_aware_prob:
                    b = np.copy(b).astype(int)
                    # box内に含まれる他のboxを考慮
                    inter_boxes = np.copy(bboxes[inter[i]])
                    inter_boxes -= np.expand_dims(
                        np.tile(b[:2], 2), axis=0
                    )  # bに合わせて平行移動
                    # random erasing
                    image[b[1] : b[3], b[0] : b[2], :] = tk.ndimage.erase_random(
                        image[b[1] : b[3], b[0] : b[2], :],
                        rand,
                        bboxes=inter_boxes,
                        scale_low=self.scale_low,
                        scale_high=self.scale_high,
                        rate_1=self.rate_1,
                        rate_2=self.rate_2,
                        max_tries=self.max_tries,
                    )
        else:
            # 画像全体でrandom erasing。
            image = tk.ndimage.erase_random(
                image,
                rand,
                bboxes=None,
                scale_low=self.scale_low,
                scale_high=self.scale_high,
                rate_1=self.rate_1,
                rate_2=self.rate_2,
                max_tries=self.max_tries,
            )
        return image

    def get_transform_init_args_names(self):
        return (
            "scale_low",
            "scale_high",
            "rate_1",
            "rate_2",
            "object_aware",
            "object_aware_prob",
            "max_tries",
        )


class GridMask(A.ImageOnlyTransform):
    """GridMask <https://arxiv.org/abs/2001.04086> <https://github.com/akuxcw/GridMask>

    - rは大きいほどマスクが小さくなる
    - d1, d2は入力サイズに対する比率で指定
    - pは(Albumentationsの流儀とは合わせず)(しかし可変にするのも面倒なので)Faster-RCNNでの実験で一番良かった0.7に

    """

    def __init__(
        self, r=0.6, d=(0.4, 1.0), random_color=False, always_apply=False, p=0.7,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.r = r
        self.d = d
        self.random_color = random_color

    def apply(self, image, **params):
        h, w = image.shape[:2]
        d = random.uniform(*self.d)
        dw = int(w * d)
        dh = int(h * d)
        lw = int(dw * self.r)
        lh = int(dh * self.r)

        # 少し大きくマスクを作成
        hh, ww = int(h * 1.5), int(w * 1.5)
        mask = np.zeros((hh, ww, 1), np.float32)
        for ox in range(0, ww, dw):
            mask[:, ox : ox + lw, :] = 1
        for oy in range(0, hh, dh):
            mask[oy : oy + lh, :, :] = 1

        # 回転
        degrees = random.uniform(0, 360)
        mask = tk.ndimage.rotate(
            mask, degrees, expand=False, interp="bilinear", border_mode="wrap"
        )

        # 使うサイズをrandom crop
        cy = random.randint(0, hh - h - 1)
        cx = random.randint(0, ww - w - 1)
        mask = mask[cy : cy + h, cx : cx + w, :]

        # マスクを適用
        if self.random_color:
            # ランダムな色でマスク (オリジナル)
            depth = 1 if image.ndim == 2 else image.shape[-1]
            color = np.array(
                [random.randint(0, 255) for _ in range(depth)], dtype=np.uint8,
            )
            return (image * mask + color * (1 - mask)).astype(image.dtype)
        else:
            return (image * mask).astype(image.dtype)

    def get_transform_init_args_names(self):
        return ("r", "d", "random_color")


class Standardize(A.ImageOnlyTransform):
    """標準化。0～255に適当に収める。"""

    def apply(self, image, **params):
        return tk.ndimage.standardize(image)

    def get_transform_init_args_names(self):
        return ()


class ToGrayScale(A.ImageOnlyTransform):
    """グレースケール化。チャンネル数はとりあえず維持。"""

    def apply(self, image, **params):
        assert len(image.shape) == 3
        start_shape = image.shape
        image = tk.ndimage.to_grayscale(image)
        image = np.tile(np.expand_dims(image, axis=-1), (1, 1, start_shape[-1]))
        assert image.shape == start_shape
        return image

    def get_transform_init_args_names(self):
        return ()


class RandomBinarize(A.ImageOnlyTransform):
    """ランダム2値化(白黒化)。"""

    def __init__(self, threshold=(127 - 32, 127 + 32), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        assert 0 < threshold[0] <= threshold[1] < 255
        self.threshold = threshold

    def apply(self, image, threshold, **params):
        return tk.ndimage.binarize(image, threshold)

    def get_params(self):
        return {"threshold": random.uniform(*self.threshold)}

    def get_transform_init_args_names(self):
        return ("threshold",)


class Binarize(A.ImageOnlyTransform):
    """2値化(白黒化)。"""

    def __init__(self, threshold=127, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)
        assert 0 < threshold < 255
        self.threshold = threshold

    def apply(self, image, **params):
        return tk.ndimage.binarize(image, self.threshold)

    def get_transform_init_args_names(self):
        return ("threshold",)


class SpeckleNoise(A.ImageOnlyTransform):
    """Speckle noise。

    References:
        - <https://github.com/tf.keras-team/tf.keras/blob/master/examples/image_ocr.py#L81>

    """

    def __init__(self, scale=(0, 15), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        rand = np.random.RandomState(random.randrange(2 ** 32))
        noise = rand.randn(*image.shape) * scale
        noise = scipy.ndimage.gaussian_filter(noise, 1)
        return np.uint8(np.clip(image + noise, 0, 255))

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class WrappedTranslateX(A.ImageOnlyTransform):
    """水平に移動してはみ出た分を反対側にくっつける。"""

    def __init__(self, scale=(-0.25, +0.25), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        scale = int(np.round(image.shape[1] * scale))
        image = np.concatenate([image[:, scale:, :], image[:, :scale, :]], axis=1)
        return image

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class WrappedTranslateY(A.ImageOnlyTransform):
    """垂直に移動してはみ出た分を反対側にくっつける。"""

    def __init__(self, scale=(-0.25, +0.25), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        scale = int(np.round(image.shape[0] * scale))
        image = np.concatenate([image[scale:, :, :], image[:scale, :, :]], axis=0)
        return image

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class To3Channel(A.ImageOnlyTransform):
    """入力がグレースケールの場合、R=G=Bにshapeを変える。"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        assert image.ndim == 3
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.shape[-1] == 3
        return image

    def get_transform_init_args_names(self):
        return ()


class To1Channel(A.ImageOnlyTransform):
    """(H, W, 1)のshapeで返す。入力がRGBの場合、(R + G + B)/3。"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        assert image.ndim == 3
        if image.shape[-1] == 3:
            image = np.mean(image, axis=-1, keepdims=True)
        assert image.shape[-1] == 1
        return image

    def get_transform_init_args_names(self):
        return ()


def _random_loguniform(lower: float, upper: float) -> float:
    """3/4 ～ 4/3みたいな乱数を作って返す。"""
    assert 0 < lower < 1 < upper
    return np.exp(random.uniform(np.log(lower), np.log(upper)))
