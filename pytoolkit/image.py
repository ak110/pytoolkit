"""画像処理関連。"""
from __future__ import annotations

import random
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

    def __call__(self, *args, force_apply=False, **data):
        """変換の適用。"""
        backup = self.transforms.transforms.copy()
        try:
            random.shuffle(self.transforms.transforms)
            return super().__call__(*args, force_apply=force_apply, **data)
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

    def apply(self, img, degrees, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.rotate(
            img, degrees, expand=self.expand, border_mode=self.border_mode
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
        mode: "normal", "preserve_aspect", "crop"

    """

    @classmethod
    def create_refine(
        cls,
        size: tuple[int, int],
        flip: tuple[bool, bool] = (False, True),
        translate: tuple[float, float] = (0.0625, 0.0625),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        mode: str = "normal",
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
            mode=mode,
            always_apply=always_apply,
            p=p,
        )

    @classmethod
    def create_test(
        cls,
        size: tuple[int, int],
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        mode: str = "normal",
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
            mode=mode,
            always_apply=always_apply,
            p=p,
        )

    def __init__(
        self,
        size: tuple[int, int],
        flip: tuple[bool, bool] = (False, True),
        translate: tuple[float, float] = (0.125, 0.125),
        scale_prob: float = 0.5,
        scale_range: tuple[float, float] = (2 / 3, 3 / 2),
        base_scale: float = 1.0,
        aspect_prob: float = 0.5,
        aspect_range: tuple[float, float] = (3 / 4, 4 / 3),
        rotate_prob: float = 0.25,
        rotate_range: tuple[int, int] = (-15, +15),
        border_mode: str = "edge",
        clip_bboxes: bool = True,
        mode: str = "normal",
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
        self.mode = mode

    def apply(self, img, m, interp=None, **params):
        # pylint: disable=arguments-differ
        cv2_border, borderValue = {
            "edge": (cv2.BORDER_REPLICATE, None),
            "reflect": (cv2.BORDER_REFLECT_101, None),
            "wrap": (cv2.BORDER_WRAP, None),
            "zero": (cv2.BORDER_CONSTANT, [0, 0, 0]),
            "half": (
                cv2.BORDER_CONSTANT,
                [0.5, 0.5, 0.5]
                if img.dtype in (np.float32, np.float64)
                else [127, 127, 127],
            ),
            "one": (
                cv2.BORDER_CONSTANT,
                [1, 1, 1] if img.dtype in (np.float32, np.float64) else [255, 255, 255],
            ),
        }[self.border_mode]

        if interp == "nearest":
            cv2_interp = cv2.INTER_NEAREST
        else:
            # 縮小ならINTER_AREA, 拡大ならINTER_LANCZOS4
            sh, sw = img.shape[:2]
            dr = cv2.perspectiveTransform(
                np.array([(0, 0), (sw, 0), (sw, sh), (0, sh)])
                .reshape((-1, 1, 2))
                .astype(np.float32),
                m,
            ).reshape((4, 2))
            dw = min(np.linalg.norm(dr[1] - dr[0]), np.linalg.norm(dr[2] - dr[3]))
            dh = min(np.linalg.norm(dr[3] - dr[0]), np.linalg.norm(dr[2] - dr[1]))
            cv2_interp = cv2.INTER_AREA if dw <= sw or dh <= sh else cv2.INTER_LANCZOS4

        if img.ndim == 2 or img.shape[-1] in (1, 3):
            img = cv2.warpPerspective(
                img,
                m,
                self.size[::-1],
                flags=cv2_interp,
                borderMode=cv2_border,
                borderValue=borderValue,
            )
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        else:
            resized_list = [
                cv2.warpPerspective(
                    img[:, :, ch],
                    m,
                    self.size[::-1],
                    flags=cv2_interp,
                    borderMode=cv2_border,
                    borderValue=borderValue,
                )
                for ch in range(img.shape[-1])
            ]
            img = np.transpose(resized_list, (1, 2, 0))
        return img

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
        img = params["image"]
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
        if self.mode == "normal":
            # アスペクト比を無視して出力サイズに合わせる
            dst_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        elif self.mode == "preserve_aspect":
            # アスペクト比を維持するように縮小する
            if img.shape[0] < img.shape[1]:
                # 横長
                hr = img.shape[0] / img.shape[1]
                yr = (1 - hr) / 2
                dst_points = np.array(
                    [[0, yr], [1, yr], [1, yr + hr], [0, yr + hr]], dtype=np.float32
                )
            else:
                # 縦長
                wr = img.shape[1] / img.shape[0]
                xr = (1 - wr) / 2
                dst_points = np.array(
                    [[xr, 0], [xr + wr, 0], [xr + wr, 1], [xr, 1]], dtype=np.float32
                )
        elif self.mode == "crop":
            # 入力サイズによらず固定サイズでcrop
            hr = self.size[0] / img.shape[0]
            wr = self.size[1] / img.shape[1]
            yr = random.uniform(0, 1 - hr)
            xr = random.uniform(0, 1 - wr)
            dst_points = np.array(
                [[xr, yr], [xr + wr, yr], [xr + wr, yr + hr], [xr, yr + hr]],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
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
        src_points[:, 0] *= img.shape[1]
        src_points[:, 1] *= img.shape[0]
        dst_points[:, 0] *= self.size[1]
        dst_points[:, 1] *= self.size[0]
        m = cv2.getPerspectiveTransform(src_points, dst_points)
        return {"m": m, "image_size": img.shape[:2]}

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

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        if self.mode == "cv2":
            img = tk.ndimage.resize(img, width=self.size[1], height=self.size[0])
        elif self.mode == "tf":
            img = tf.image.resize(
                img, self.size, method=tf.image.ResizeMethod.LANCZOS5, antialias=True
            )
            img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()
        elif self.mode == "pil":
            if img.ndim == 3 and img.shape[2] == 1:
                img = np.squeeze(img, axis=2)
            img = np.asarray(
                PIL.Image.fromarray(img).resize(
                    self.size[::-1], resample=PIL.Image.LANCZOS
                ),
                dtype=np.uint8,
            )
            img = tk.ndimage.ensure_channel_dim(img)
        else:
            raise ValueError(f"mode={self.mode}")
        return img

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

    def apply(self, img, scale, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.getrandbits(32))
        return tk.ndimage.gaussian_noise(img, rand, scale)

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

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.getrandbits(32))
        map_ = rand.uniform(size=img.shape[:2])
        img = img.copy()
        img[map_ < self.salt] = 255
        img[map_ > 1 - self.pepper] = 0
        return img

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

    def apply(self, img, alpha, frequency, octaves, ar, seed, **params):
        # pylint: disable=arguments-differ
        noise = PerlinNoise.perlin_noise(
            img.shape[:2], frequency=frequency, octaves=octaves, ar=ar, seed=seed
        )
        if img.ndim != 2:
            assert img.ndim == 3
            noise = np.expand_dims(noise, axis=-1)
        img = (img * (1 - alpha) + noise * alpha).astype(img.dtype)
        return img

    @staticmethod
    def perlin_noise(
        shape: tuple[int, int],
        frequency: float = 3.0,
        octaves: int = 5,
        ar: float = 1.0,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int = 0,
    ):
        """パーリンノイズの生成。

        Args:
            shape: (H, W)
            frequency: 短辺の周波数
            octaves: 重ねる数
            ar: アスペクト比。>1なら横長、<1なら縦長のノイズを生成する

        """
        random_state = np.random.RandomState(seed=seed)
        rar = np.sqrt(ar)
        if shape[0] < shape[1]:
            freq_v = frequency / rar
            freq_h = frequency * shape[1] / shape[0] * rar
        else:
            freq_h = frequency * rar
            freq_v = frequency * shape[0] / shape[1] / rar
        grad_size_h = int(freq_h * lacunarity ** octaves + 1)
        grad_size_v = int(freq_v * lacunarity ** octaves + 1)
        gradient = random_state.uniform(-1, +1, size=(grad_size_h, grad_size_v, 2))

        img = np.zeros(shape, dtype=np.float32)
        amplify = 1.0
        for _ in range(octaves):
            img += (
                PerlinNoise._perlin_noise_base(shape, freq_v, freq_h, gradient)
                * amplify
            )
            freq_v *= lacunarity
            freq_h *= lacunarity
            amplify *= persistence
        img -= img.min()
        img /= img.max()
        return np.uint8(img * 255)

    @staticmethod
    def _perlin_noise_base(shape, freq_v, freq_h, gradient):
        # linear space by frequency
        x = np.tile(np.linspace(0, freq_h, shape[1], endpoint=False), shape[0])
        y = np.repeat(np.linspace(0, freq_v, shape[0], endpoint=False), shape[1])

        # gradient coordinates
        x0 = x.astype(int)
        y0 = y.astype(int)

        # local coordinate
        x -= x0
        y -= y0

        # gradient projections
        g00 = gradient[x0, y0]
        g10 = gradient[x0 + 1, y0]
        g01 = gradient[x0, y0 + 1]
        g11 = gradient[x0 + 1, y0 + 1]

        # fade
        t = (3 - 2 * x) * x * x

        # linear interpolation
        r = g00[:, 0] * x + g00[:, 1] * y
        s = g10[:, 0] * (x - 1) + g10[:, 1] * y
        g0 = r + t * (s - r)

        # linear interpolation
        r = g01[:, 0] * x + g01[:, 1] * (y - 1)
        s = g11[:, 0] * (x - 1) + g11[:, 1] * (y - 1)
        g1 = r + t * (s - r)

        # fade
        t = (3 - 2 * y) * y * y

        # (bi)linear interpolation
        g = g0 + t * (g1 - g0)

        # reshape
        return g.reshape(shape)

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

    def apply(self, img, sigma, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.blur(img, sigma)

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

    def apply(self, img, alpha, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.unsharp_mask(img, self.sigma, alpha)

    def get_params(self):
        return {"alpha": random.uniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("sigma", "alpha")


class RandomBrightness(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """明度の変更。"""

    def __init__(self, shift=(-50, 50), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.shift = shift

    def apply(self, img, shift, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.brightness(img, shift)

    def get_params(self):
        return {"shift": random.uniform(*self.shift)}

    def get_transform_init_args_names(self):
        return ("shift",)


class RandomContrast(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """コントラストの変更。"""

    def __init__(self, alpha=(1 / 2, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img, alpha, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.contrast(img, alpha)

    def get_params(self):
        return {"alpha": _random_loguniform(*self.alpha)}

    def get_transform_init_args_names(self):
        return ("alpha",)


class RandomSaturation(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """彩度の変更。"""

    def __init__(self, alpha=(1 / 2, 2), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha

    def apply(self, img, alpha, **params):
        # pylint: disable=arguments-differ
        if img.shape[-1] != 3:
            return img
        return tk.ndimage.saturation(img, alpha)

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

    def apply(self, img, alpha, beta, **params):
        # pylint: disable=arguments-differ
        if img.shape[-1] != 3:
            return img
        return tk.ndimage.hue_lite(img, alpha, beta)

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

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.equalize(img)

    def get_transform_init_args_names(self):
        return ()


class RandomAutoContrast(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """オートコントラスト。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.auto_contrast(img)

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

    def apply(self, img, bits, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.posterize(img, bits)

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

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.getrandbits(32))
        return tk.ndimage.erase_random(
            img,
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

    def apply(self, img, bboxes=None, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.getrandbits(32))
        img = np.copy(img)
        bboxes = (
            np.round(bboxes * np.array(img.shape)[[1, 0, 1, 0]])
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
                    img[b[1] : b[3], b[0] : b[2], :] = tk.ndimage.erase_random(
                        img[b[1] : b[3], b[0] : b[2], :],
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
            img = tk.ndimage.erase_random(
                img,
                rand,
                bboxes=None,
                scale_low=self.scale_low,
                scale_high=self.scale_high,
                rate_1=self.rate_1,
                rate_2=self.rate_2,
                max_tries=self.max_tries,
            )
        return img

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
        r: float | tuple[float, float] = 0.6,
        d: tuple[float, float] = (0.4, 1.0),
        random_color: bool = False,
        fill_value: int = 0,
        always_apply: bool = False,
        p: float = 0.7,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.r: tuple[float, float] = r if isinstance(r, tuple) else (r, r)
        self.d: tuple[float, float] = d
        self.random_color = random_color
        self.fill_value = fill_value

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        # ↓<https://github.com/PyCQA/pylint/issues/3139>
        # pylint: disable=unsubscriptable-object
        h, w = img.shape[:2]
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
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        if self.random_color:
            # ランダムな色でマスク (オリジナル)
            color = np.array(
                [random.randint(0, 255) for _ in range(img.shape[-1])], dtype=np.uint8
            )
            return (img * mask + color * (1 - mask)).astype(img.dtype)
        elif self.fill_value != 0:
            color = np.asarray(self.fill_value)
            return (img * mask + color * (1 - mask)).astype(img.dtype)
        else:
            return (img * mask).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("r", "d", "random_color")


class Standardize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """標準化。0～255に適当に収める。"""

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.standardize(img)

    def get_transform_init_args_names(self):
        return ()


class ToGrayScale(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """グレースケール化。チャンネル数はとりあえず維持。"""

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        assert len(img.shape) == 3
        start_shape = img.shape
        img = tk.ndimage.to_grayscale(img)
        img = np.tile(np.expand_dims(img, axis=-1), (1, 1, start_shape[-1]))
        assert img.shape == start_shape
        return img

    def get_transform_init_args_names(self):
        return ()


class RandomBinarize(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """ランダム2値化(白黒化)。"""

    def __init__(self, threshold=(127 - 32, 127 + 32), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        assert 0 < threshold[0] <= threshold[1] < 255
        self.threshold = threshold

    def apply(self, img, threshold, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.binarize(img, threshold)

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

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        return tk.ndimage.binarize(img, self.threshold)

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

    def apply(self, img, scale, **params):
        # pylint: disable=arguments-differ
        rand = np.random.RandomState(random.getrandbits(32))
        noise = rand.randn(*img.shape) * scale
        noise = scipy.ndimage.gaussian_filter(noise, 1)
        return np.uint8(np.clip(img + noise, 0, 255))

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
        - <https://www.kaggle.com/ren4yu/bengali-morphological-ops-as-img-augmentation>
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

    def apply(self, img, **params):  # pylint: disable=arguments-differ
        ksize = random.randint(*self.ksize), random.randint(*self.ksize)
        kernel = cv2.getStructuringElement(self.element_shape, ksize)

        if self.mode == "erode":
            img = cv2.erode(img, kernel, iterations=1)
        elif self.mode == "dilate":
            img = cv2.dilate(img, kernel, iterations=1)
        elif self.mode == "open":
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif self.mode == "close":
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img

    def get_transform_init_args_names(self):
        return ("mode", "ksize", "element_shape")


class WrappedTranslateX(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """水平に移動してはみ出た分を反対側にくっつける。"""

    def __init__(self, scale=(-0.25, +0.25), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, img, scale, **params):
        # pylint: disable=arguments-differ
        scale = int(np.round(img.shape[1] * scale))
        img = np.concatenate([img[:, scale:, :], img[:, :scale, :]], axis=1)
        return img

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class WrappedTranslateY(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """垂直に移動してはみ出た分を反対側にくっつける。"""

    def __init__(self, scale=(-0.25, +0.25), always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, img, scale, **params):
        # pylint: disable=arguments-differ
        scale = int(np.round(img.shape[0] * scale))
        img = np.concatenate([img[scale:, :, :], img[:scale, :, :]], axis=0)
        return img

    def get_params(self):
        return {"scale": random.uniform(*self.scale)}

    def get_transform_init_args_names(self):
        return ("scale",)


class To3Channel(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """入力がグレースケールの場合、R=G=Bにshapeを変える。"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        assert img.ndim == 3
        if img.shape[-1] == 1:
            img = np.tile(img, (1, 1, 3))
        assert img.shape[-1] == 3
        return img

    def get_transform_init_args_names(self):
        return ()


class To1Channel(A.ImageOnlyTransform):  # pylint: disable=abstract-method
    """(H, W, 1)のshapeで返す。入力がRGBの場合、(R + G + B)/3。"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # pylint: disable=arguments-differ
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        assert img.ndim == 3
        if img.shape[-1] == 3:
            img = np.mean(img, axis=-1, keepdims=True)
        assert img.shape[-1] == 1
        return img

    def get_transform_init_args_names(self):
        return ()


def _random_loguniform(lower: float, upper: float) -> float:
    """3/4 ～ 4/3みたいな乱数を作って返す。"""
    assert 0 < lower < 1 < upper
    return np.exp(random.uniform(np.log(lower), np.log(upper)))
