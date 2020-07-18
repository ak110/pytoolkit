"""画像処理関連。"""
from __future__ import annotations

import random
import typing
import warnings

import albumentations as A
import cv2
import numpy as np
import PIL.Image
import scipy.ndimage
import tensorflow as tf

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
        # pylint: disable=arguments-differ
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
    """Flip, Scale, Resize, Rotateをまとめて処理。

    Args:
        size: 出力サイズ(h, w)
        flip: 反転の有無(vertical, horizontal)
        translate: 平行移動の量(vertical, horizontal)
        border_mode: edge, reflect, wrap, zero, half, one
        preserve_aspect: アスペクト比を維持するように縮小するならTrue

    """

    @classmethod
    def create_refine(
        cls,
        size: typing.Tuple[int, int],
        flip: typing.Tuple[bool, bool] = (False, True),
        translate: typing.Tuple[float, float] = (0.0625, 0.0625),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        preserve_aspect: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> RandomTransform:
        """Refined Data Augmentation <https://arxiv.org/abs/1909.09148> 用の控えめバージョンを作成する。"""
        return cls(
            size=size,
            flip=flip,
            translate=translate,
            scale_prob=0.0,
            aspect_prob=0.0,
            rotate_prob=0.0,
            border_mode=border_mode,
            clip_bboxes=clip_bboxes,
            preserve_aspect=preserve_aspect,
            always_apply=always_apply,
            p=p,
        )

    @classmethod
    def create_test(
        cls,
        size: typing.Tuple[int, int],
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        preserve_aspect: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> RandomTransform:
        """Data Augmentation無しバージョン(リサイズのみ)を作成する。"""
        return cls(
            size=size,
            flip=(False, False),
            translate=(0.0, 0.0),
            scale_prob=0.0,
            aspect_prob=0.0,
            rotate_prob=0.0,
            border_mode=border_mode,
            clip_bboxes=clip_bboxes,
            preserve_aspect=preserve_aspect,
            always_apply=always_apply,
            p=p,
        )

    def __init__(
        self,
        size,
        flip: typing.Tuple[bool, bool] = (False, True),
        translate: typing.Tuple[float, float] = (0.125, 0.125),
        scale_prob: float = 0.5,
        scale_range: typing.Tuple[float, float] = (2 / 3, 3 / 2),
        base_scale: float = 1.0,
        aspect_prob: float = 0.5,
        aspect_range: typing.Tuple[float, float] = (3 / 4, 4 / 3),
        rotate_prob: float = 0.25,
        rotate_range: typing.Tuple[int, int] = (-15, +15),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        preserve_aspect: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.size = size
        self.flip = flip
        self.translate = translate
        self.scale_prob = scale_prob
        self.base_scale = base_scale
        self.scale_range = scale_range
        self.aspect_prob = aspect_prob
        self.aspect_range = aspect_range
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.border_mode = border_mode
        self.clip_bboxes = clip_bboxes
        self.preserve_aspect = preserve_aspect

    def apply(self, image, m, interp=None, **params):
        # pylint: disable=arguments-differ
        cv2_border, borderValue = {
            "edge": (cv2.BORDER_REPLICATE, None),
            "reflect": (cv2.BORDER_REFLECT_101, None),
            "wrap": (cv2.BORDER_WRAP, None),
            "zero": (cv2.BORDER_CONSTANT, [0, 0, 0]),
            "half": (cv2.BORDER_CONSTANT, [127, 127, 127]),
            "one": (cv2.BORDER_CONSTANT, [255, 255, 255]),
        }[self.border_mode]

        if interp == "nearest":
            cv2_interp = cv2.INTER_NEAREST
        else:
            # 縮小ならINTER_AREA, 拡大ならINTER_LANCZOS4
            sh, sw = image.shape[:2]
            dr = cv2.perspectiveTransform(
                np.array([(0, 0), (sw, 0), (sw, sh), (0, sh)])
                .reshape((-1, 1, 2))
                .astype(np.float32),
                m,
            ).reshape((4, 2))
            dw = min(np.linalg.norm(dr[1] - dr[0]), np.linalg.norm(dr[2] - dr[3]))
            dh = min(np.linalg.norm(dr[3] - dr[0]), np.linalg.norm(dr[2] - dr[1]))
            cv2_interp = cv2.INTER_AREA if dw <= sw or dh <= sh else cv2.INTER_LANCZOS4

        if image.ndim == 2 or image.shape[-1] in (1, 3):
            image = cv2.warpPerspective(
                image,
                m,
                self.size[::-1],
                flags=cv2_interp,
                borderMode=cv2_border,
                borderValue=borderValue,
            )
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
        else:
            resized_list = [
                cv2.warpPerspective(
                    image[:, :, ch],
                    m,
                    self.size[::-1],
                    flags=cv2_interp,
                    borderMode=cv2_border,
                    borderValue=borderValue,
                )
                for ch in range(image.shape[-1])
            ]
            image = np.transpose(resized_list, (1, 2, 0))
        return image

    def apply_to_bbox(self, bbox, m, image_size, **params):
        # pylint: disable=arguments-differ
        del params
        bbox = np.asarray(bbox)
        assert bbox.shape == (4,)
        bbox *= np.array([image_size[1], image_size[0]] * 2)
        bbox = cv2.perspectiveTransform(
            bbox.reshape((-1, 1, 2)).astype(np.float32), m
        ).reshape(bbox.shape)
        if bbox[2] < bbox[0]:
            bbox = bbox[[2, 1, 0, 3]]
        if bbox[3] < bbox[1]:
            bbox = bbox[[0, 3, 2, 1]]
        bbox /= np.array([self.size[1], self.size[0]] * 2)
        assert bbox.shape == (4,)
        if self.clip_bboxes:
            bbox = np.clip(bbox, 0, 1)
        return tuple(bbox)

    def apply_to_keypoint(self, keypoint, m, **params):
        # pylint: disable=arguments-differ
        del params
        xy = np.asarray(keypoint[:2])
        xy = cv2.perspectiveTransform(
            xy.reshape((-1, 1, 2)).astype(np.float32), m
        ).reshape(xy.shape)
        return tuple(xy) + tuple(keypoint[2:])

    def apply_to_mask(self, img, interp=None, **params):
        # pylint: disable=arguments-differ
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

        flip_v = self.flip[0] and random.random() <= 0.5
        flip_h = self.flip[1] and random.random() <= 0.5
        scale_v = scale / np.sqrt(ar)
        scale_h = scale * np.sqrt(ar)
        degrees = (
            random.uniform(self.rotate_range[0], self.rotate_range[1])
            if random.random() <= self.rotate_prob
            else 0
        )
        pos_v = random.uniform(0, 1)
        pos_h = random.uniform(0, 1)
        translate_v = random.uniform(-self.translate[0], self.translate[0])
        translate_h = random.uniform(-self.translate[1], self.translate[1])
        # 左上から時計回りに座標を用意
        src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        if self.preserve_aspect:
            # アスペクト比を維持するように縮小する
            if image.shape[0] < image.shape[1]:
                # 横長
                hr = image.shape[0] / image.shape[1]
                yr = (1 - hr) / 2
                dst_points = np.array(
                    [[0, yr], [1, yr], [1, yr + hr], [0, yr + hr]], dtype=np.float32
                )
            else:
                # 縦長
                wr = image.shape[1] / image.shape[0]
                xr = (1 - wr) / 2
                dst_points = np.array(
                    [[xr, 0], [xr + wr, 0], [xr + wr, 1], [xr, 1]], dtype=np.float32
                )
        else:
            # アスペクト比を無視して出力サイズに合わせる
            dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        # 反転
        if flip_h:
            dst_points = dst_points[[1, 0, 3, 2]]
        if flip_v:
            dst_points = dst_points[[3, 2, 1, 0]]
        # 移動
        src_points[:, 0] -= translate_h
        src_points[:, 1] -= translate_v
        # 回転
        theta = degrees * np.pi * 2 / 360
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]], dtype=np.float32)
        src_points = np.dot(r, (src_points - 0.5).T).T + 0.5
        # スケール変換
        src_points[:, 0] /= scale_h
        src_points[:, 1] /= scale_v
        src_points[:, 0] -= (1 / scale_h - 1) * pos_h
        src_points[:, 1] -= (1 / scale_v - 1) * pos_v
        # 変換行列の作成
        src_points[:, 0] *= image.shape[1]
        src_points[:, 1] *= image.shape[0]
        dst_points[:, 0] *= self.size[1]
        dst_points[:, 1] *= self.size[0]
        m = cv2.getPerspectiveTransform(src_points, dst_points)
        return {"m": m, "image_size": image.shape[:2]}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "size",
            "flip",
            "translate",
            "scale_prob",
            "scale_range",
            "base_scale",
            "aspect_prob",
            "aspect_range",
            "rotate_prob",
            "rotate_range",
            "border_mode",
        )


class Resize(A.DualTransform):  # pylint: disable=abstract-method
    """リサイズ。

    Args:
        size: 出力サイズ。(H, W)
        mode: cv2 or tf or pil

    環境にもよるが、1024x768 -> 331x331でcv2:3ms、tf:30ms、pil:10msくらい。
    (tf/pilは遅めなので要注意。ただしcv2は拡大と縮小の混在は画質が悪いので要注意。)

    """

    def __init__(
        self, *, size=None, width=None, height=None, mode="cv2", always_apply=False, p=1
    ):
        super().__init__(always_apply=False, p=1)
        if size is None:
            # deprecated
            warnings.warn("width/height ware deprecated.")
            size = (height, width)
        assert mode in ("cv2", "tf", "pil")
        self.size = size
        self.mode = mode

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        if self.mode == "cv2":
            image = tk.ndimage.resize(image, width=self.size[1], height=self.size[0])
        elif self.mode == "tf":
            image = tf.image.resize(
                image, self.size, method=tf.image.ResizeMethod.LANCZOS5, antialias=True
            )
            image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8).numpy()
        elif self.mode == "pil":
            if image.ndim == 3 and image.shape[2] == 1:
                image = np.squeeze(image, axis=2)
            image = np.asarray(
                PIL.Image.fromarray(image).resize(
                    self.size[::-1], resample=PIL.Image.LANCZOS
                ),
                dtype=np.uint8,
            )
            image = tk.ndimage.ensure_channel_dim(image)
        else:
            raise ValueError(f"mode={self.mode}")
        return image

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("size",)


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
            RandomAlpha(p=0.25),
        ]
        if noisy:
            argumentors.extend(
                [
                    RandomEqualize(p=0.0625),
                    RandomAutoContrast(p=0.0625),
                    RandomPosterize(p=0.0625),
                    A.Solarize(threshold=(50, 255 - 50), p=0.0625),
                    RandomBlur(p=0.125),
                    RandomUnsharpMask(p=0.125),
                    A.IAASharpen(alpha=(0, 0.5), p=0.125),
                    GaussNoise(p=0.125),
                ]
            )
        if not grayscale and noisy:
            argumentors.extend(
                [A.ISONoise(color_shift=(0, 0.05), intensity=(0, 0.5), p=0.125)]
            )
        super().__init__(argumentors, p=p)

    def get_transform_init_args_names(self):
        return ("noisy", "grayscale")


class GaussNoise(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """ガウシアンノイズ。"""

    def __init__(self, scale=(0, 15), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.randrange(2 ** 32))
        return tk.ndimage.gaussian_noise(image, rand, scale)

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class SaltAndPepperNoise(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """Salt-and-pepper noise。"""

    def __init__(self, salt=0.01, pepper=0.01, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.salt = salt
        self.pepper = pepper

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
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


class PerlinNoise(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """Perlin noise。"""

    def __init__(
        self,
        alpha=(0, 0.5),
        frequency=(2.0, 4.0),
        octaves=(3, 5),
        ar=(0.5, 2.0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.frequency = frequency
        self.octaves = octaves
        self.ar = ar

    def apply(self, image, alpha, frequency, octaves, ar, seed, **params):
        # pylint: disable=arguments-differ
        noise = tk.ndimage.perlin_noise(
            image.shape[:2], frequency=frequency, octaves=octaves, ar=ar, seed=seed
        )
        if image.ndim != 2:
            assert image.ndim == 3
            noise = np.expand_dims(noise, axis=-1)
        image = (image * (1 - alpha) + noise * alpha).astype(image.dtype)
        return image

    def get_params(self):
        return {
            "alpha": random.uniform(*self.alpha),
            "frequency": random.uniform(*self.frequency),
            "octaves": random.randint(*self.octaves),
            "ar": np.exp(random.uniform(np.log(self.ar[0]), np.log(self.ar[1]))),
            "seed": random.randint(0, 2 ** 32 - 1),
        }

    def get_transform_init_args_names(self):
        return ("alpha", "frequency", "octaves", "ar")


class RandomBlur(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """ぼかし。"""

    def __init__(self, sigma=(0, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma

    def apply(self, image, sigma, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.blur(image, sigma)

    def get_params(self):
        return {"sigma": random.uniform(*self.sigma)}

    def get_transform_init_args_names(self):
        return ("sigma",)


class RandomUnsharpMask(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """シャープ化。"""

    def __init__(self, sigma=0.5, alpha=(0, 3), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma
        self.alpha = alpha

    def apply(self, image, alpha, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.unsharp_mask(image, self.sigma, alpha)

    def get_params(self):
        return {"alpha": random.uniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("sigma", "alpha")


class RandomBrightness(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """明度の変更。"""

    def __init__(self, shift=(-50, 50), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.shift = shift

    def apply(self, image, shift, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.brightness(image, shift)

    def get_params(self):
        return {"shift": random.uniform(*self.shift)}

    def get_transform_init_args_names(self):
        return ("shift",)


class RandomContrast(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """コントラストの変更。"""

    def __init__(self, alpha=(1 / 2, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, image, alpha, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.contrast(image, alpha)

    def get_params(self):
        return {"alpha": _random_loguniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)


class RandomSaturation(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """彩度の変更。"""

    def __init__(self, alpha=(1 / 2, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, image, alpha, **params):
        # pylint: disable=arguments-differ
        if image.shape[-1] != 3:
            return image
        return tk.ndimage.saturation(image, alpha)

    def get_params(self):
        return {"alpha": _random_loguniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)


class RandomHue(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """色相の変更。"""

    def __init__(self, alpha=(1 / 1.5, 1.5), beta=(-30, 30), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.beta = beta

    def apply(self, image, alpha, beta, **params):
        # pylint: disable=arguments-differ
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


class RandomEqualize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """ヒストグラム平坦化。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.equalize(image)

    def get_transform_init_args_names(self):
        return ()


class RandomAutoContrast(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """オートコントラスト。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.auto_contrast(image)

    def get_transform_init_args_names(self):
        return ()


class RandomPosterize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """ポスタリゼーション。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def __init__(self, bits=(4, 7), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.bits = bits

    def apply(self, image, bits, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.posterize(image, bits)

    def get_params(self):
        return {"bits": random.randint(*self.bits)}

    def get_transform_init_args_names(self):
        return ("bits",)


class RandomAlpha(A.ImageOnlyTransform):  # pylint: disable=abstract-method
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
        # pylint: disable=arguments-differ
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


class RandomErasing(A.ImageOnlyTransform):  # pylint: disable=abstract-method
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
        # pylint: disable=arguments-differ
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


class GridMask(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """GridMask <https://arxiv.org/abs/2001.04086> <https://github.com/akuxcw/GridMask>

    - rは大きいほどマスクが小さくなる
    - d1, d2は入力サイズに対する比率で指定
    - pは(Albumentationsの流儀とは合わせず)(しかし可変にするのも面倒なので)Faster-RCNNでの実験で一番良かった0.7に

    """

    def __init__(
        self,
        r: typing.Union[float, typing.Tuple[float, float]] = 0.6,
        d: typing.Tuple[float, float] = (0.4, 1.0),
        random_color: bool = False,
        fill_value: int = 0,
        always_apply: bool = False,
        p: float = 0.7,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.r: typing.Tuple[float, float] = r if isinstance(r, tuple) else (r, r)
        self.d: typing.Tuple[float, float] = d
        self.random_color = random_color
        self.fill_value = fill_value

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        # ↓<https://github.com/PyCQA/pylint/issues/3139>
        # pylint: disable=unsubscriptable-object
        h, w = image.shape[:2]
        d = int(min(h, w) * random.uniform(*self.d))
        r = random.uniform(*self.r)
        l_ = int(d * r)

        # 少し大きくマスクを作成
        hh, ww = int(h * 1.5), int(w * 1.5)
        mask = np.zeros((hh, ww, 1), np.float32)
        for ox in range(0, ww, d):
            mask[:, ox : ox + l_, :] = 1
        for oy in range(0, hh, d):
            mask[oy : oy + l_, :, :] = 1

        # 回転
        degrees = random.uniform(0, 360)
        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        m = cv2.getRotationMatrix2D(center=center, angle=degrees, scale=1.0)
        mask = cv2.warpAffine(
            mask,
            m,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_WRAP,
        )
        assert mask.ndim == 2
        mask = np.expand_dims(mask, axis=-1)

        # 使うサイズをrandom crop
        cy = random.randint(0, hh - h - 1)
        cx = random.randint(0, ww - w - 1)
        mask = mask[cy : cy + h, cx : cx + w, :]

        # マスクを適用
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if self.random_color:
            # ランダムな色でマスク (オリジナル)
            color = np.array(
                [random.randint(0, 255) for _ in range(image.shape[-1])],
                dtype=np.uint8,
            )
            return (image * mask + color * (1 - mask)).astype(image.dtype)
        elif self.fill_value != 0:
            color = np.asarray(self.fill_value)
            return (image * mask + color * (1 - mask)).astype(image.dtype)
        else:
            return (image * mask).astype(image.dtype)

    def get_transform_init_args_names(self):
        return ("r", "d", "random_color")


class Standardize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """標準化。0～255に適当に収める。"""

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.standardize(image)

    def get_transform_init_args_names(self):
        return ()


class ToGrayScale(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """グレースケール化。チャンネル数はとりあえず維持。"""

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        assert len(image.shape) == 3
        start_shape = image.shape
        image = tk.ndimage.to_grayscale(image)
        image = np.tile(np.expand_dims(image, axis=-1), (1, 1, start_shape[-1]))
        assert image.shape == start_shape
        return image

    def get_transform_init_args_names(self):
        return ()


class RandomBinarize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """ランダム2値化(白黒化)。"""

    def __init__(self, threshold=(127 - 32, 127 + 32), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        assert 0 < threshold[0] <= threshold[1] < 255
        self.threshold = threshold

    def apply(self, image, threshold, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.binarize(image, threshold)

    def get_params(self):
        return {"threshold": random.uniform(*self.threshold)}

    def get_transform_init_args_names(self):
        return ("threshold",)


class Binarize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """2値化(白黒化)。"""

    def __init__(self, threshold=127, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)
        assert 0 < threshold < 255
        self.threshold = threshold

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.binarize(image, self.threshold)

    def get_transform_init_args_names(self):
        return ("threshold",)


class SpeckleNoise(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """Speckle noise。

    References:
        - <https://github.com/tf.keras-team/tf.keras/blob/master/examples/image_ocr.py#L81>

    """

    def __init__(self, scale=(0, 15), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.randrange(2 ** 32))
        noise = rand.randn(*image.shape) * scale
        noise = scipy.ndimage.gaussian_filter(noise, 1)
        return np.uint8(np.clip(image + noise, 0, 255))

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class RandomMorphology(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """モルフォロジー変換

    Args:
        mode: 動作モード
            - "erode"
            - "dilate"
            - "open"
            - "close"
        ksize: カーネルサイズの幅 (min, max)
        element_shape: カーネルの形状
            - cv2.MORPH_ELLIPSE: 楕円
            - cv2.MORPH_RECT: 矩形

    References:
        - <https://www.kaggle.com/ren4yu/bengali-morphological-ops-as-image-augmentation>
        - <https://www.kaggle.com/c/bengaliai-cv19/discussion/128198#734220>
        - <http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html>  # noqa

    """

    def __init__(
        self,
        mode,
        ksize=(1, 5),
        element_shape=cv2.MORPH_ELLIPSE,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        assert mode in ("erode", "dilate", "open", "close")
        self.mode = mode
        self.ksize = ksize
        self.element_shape = element_shape

    def apply(self, image, **params):  # pylint: disable=arguments-differ
        ksize = random.randint(*self.ksize), random.randint(*self.ksize)
        kernel = cv2.getStructuringElement(self.element_shape, ksize)

        if self.mode == "erode":
            image = cv2.erode(image, kernel, iterations=1)
        elif self.mode == "dilate":
            image = cv2.dilate(image, kernel, iterations=1)
        elif self.mode == "open":
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif self.mode == "close":
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        return image

    def get_transform_init_args_names(self):
        return ("mode", "ksize", "element_shape")


class WrappedTranslateX(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """水平に移動してはみ出た分を反対側にくっつける。"""

    def __init__(self, scale=(-0.25, +0.25), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        # pylint: disable=arguments-differ
        scale = int(np.round(image.shape[1] * scale))
        image = np.concatenate([image[:, scale:, :], image[:, :scale, :]], axis=1)
        return image

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class WrappedTranslateY(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """垂直に移動してはみ出た分を反対側にくっつける。"""

    def __init__(self, scale=(-0.25, +0.25), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, scale, **params):
        # pylint: disable=arguments-differ
        scale = int(np.round(image.shape[0] * scale))
        image = np.concatenate([image[scale:, :, :], image[:scale, :, :]], axis=0)
        return image

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class To3Channel(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """入力がグレースケールの場合、R=G=Bにshapeを変える。"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        assert image.ndim == 3
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.shape[-1] == 3
        return image

    def get_transform_init_args_names(self):
        return ()


class To1Channel(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """(H, W, 1)のshapeで返す。入力がRGBの場合、(R + G + B)/3。"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        # pylint: disable=arguments-differ
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
