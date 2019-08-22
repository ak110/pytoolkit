"""画像処理関連。

Albumentations風インターフェースだが、seedの固定とかがしたいので独自実装。

"""
# pylint: disable=signature-differs,arguments-differ,unused-argument
import abc
import warnings

import numpy as np
import sklearn.utils

import pytoolkit as tk


class BasicTransform(metaclass=abc.ABCMeta):
    """変換を行うクラス。"""

    def __init__(self, always_apply=False, p=1.0):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, **data):
        data["random"] = sklearn.utils.check_random_state(data.get("random", None))
        if data["random"].rand() <= self.p:
            data = self.apply_transform(**data)
        return data

    @abc.abstractmethod
    def apply_transform(self, **data):
        """変換の適用。"""


class BasicCompose(BasicTransform, metaclass=abc.ABCMeta):
    """複数の変換をまとめて適用するクラス。"""

    def __init__(self, transforms, p=1.0):
        super().__init__(p=p)
        self.transforms = transforms

    @abc.abstractmethod
    def apply_transform(self, **data):
        """変換の適用。"""


class Compose(BasicCompose):
    """複数の変換をまとめて適用するクラス。"""

    def apply_transform(self, **data):
        """変換の適用。"""
        for transform in self.transforms:
            data = transform(**data)
        return data


class RandomCompose(Compose):
    """シャッフル付きCompose。"""

    def apply_transform(self, **data):
        """変換の適用。"""
        backup = self.transforms.copy()
        try:
            data["random"].shuffle(self.transforms)
            return super().apply_transform(**data)
        finally:
            self.transforms = backup


class OneOf(BasicCompose):
    """複数の中からランダムに1つだけ適用するクラス。"""

    def apply_transform(self, **data):
        """変換の適用。"""
        transform = data["random"].choice(self.transforms)
        data = transform(**data)
        return data


class ImageOnlyTransform(BasicTransform):
    """画像のみの変換."""

    @property
    def targets(self):
        return {"image": self.apply}

    def apply_transform(self, **data):
        """変換の適用。"""
        targets = self.targets
        params = {key: value for key, value in data.items() if key not in targets}
        params.update(self.get_params(**data))
        result = {}
        for key, value in data.items():
            if key in targets:
                result[key] = targets[key](value, **params)
            else:
                result[key] = value
        return result

    def apply(self, image, **params):
        """画像の変換。"""
        raise NotImplementedError(
            "Method apply is not implemented in class " + self.__class__.__name__
        )

    def get_params(self, **data):
        """パラメータを返す。"""
        return {}


class DualTransform(ImageOnlyTransform):
    """画像、バウンディングボックス、キーポイント、マスクの変換。"""

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox, **params):
        """バウンディングボックス(単数)の変換。"""
        raise NotImplementedError(
            "Method apply_to_bbox is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_keypoint(self, keypoint, **params):
        """キーポイント(単数)の変換。"""
        raise NotImplementedError(
            "Method apply_to_keypoint is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_mask(self, image, **params):
        """マスク(単数)の変換。"""
        return self.apply(image, **params)

    def apply_to_bboxes(self, bboxes, **params):
        """バウンディングボックス(複数)の変換。"""
        bboxes = [list(bbox) for bbox in bboxes]
        return [self.apply_to_bbox(bbox[:4], **params) + bbox[4:] for bbox in bboxes]

    def apply_to_keypoints(self, keypoints, **params):
        """キーポイント(複数)の変換。"""
        keypoints = [list(keypoint) for keypoint in keypoints]
        return [
            self.apply_to_keypoint(keypoint[:4], **params) + keypoint[4:]
            for keypoint in keypoints
        ]

    def apply_to_masks(self, masks, **params):
        """マスク(複数)の変換。"""
        return [self.apply_to_mask(mask, **params) for mask in masks]


class RandomRotate(DualTransform):
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

    def get_params(self, **data):
        return {"degrees": data["random"].uniform(-self.degrees, self.degrees)}


class RandomTransform(DualTransform):
    """Flip, Scale, Resize, Rotateをまとめて処理。"""

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

    def apply(
        self,
        image,
        flip_h,
        flip_v,
        scale_h,
        scale_v,
        degrees,
        pos_h,
        pos_v,
        translate_h,
        translate_v,
        **params,
    ):
        m = tk.ndimage.compute_perspective(
            image.shape[1],
            image.shape[0],
            self.width,
            self.height,
            flip_h=flip_h,
            flip_v=flip_v,
            scale_h=scale_h,
            scale_v=scale_v,
            degrees=degrees,
            pos_h=pos_h,
            pos_v=pos_v,
            translate_h=translate_h,
            translate_v=translate_v,
        )
        return tk.ndimage.perspective_transform(
            image,
            self.width,
            self.height,
            m,
            interp=self.interp,
            border_mode=self.border_mode,
        )

    def apply_to_bbox(
        self,
        bbox,
        image,
        flip_h,
        flip_v,
        scale_h,
        scale_v,
        degrees,
        pos_h,
        pos_v,
        translate_h,
        translate_v,
        **params,
    ):
        m = tk.ndimage.compute_perspective(
            image.shape[1],
            image.shape[0],
            self.width,
            self.height,
            flip_h=flip_h,
            flip_v=flip_v,
            scale_h=scale_h,
            scale_v=scale_v,
            degrees=degrees,
            pos_h=pos_h,
            pos_v=pos_v,
            translate_h=translate_h,
            translate_v=translate_v,
        )
        raise tk.ndimage.transform_points(bbox, m)

    def apply_to_keypoint(
        self,
        keypoint,
        image,
        flip_h,
        flip_v,
        scale_h,
        scale_v,
        degrees,
        pos_h,
        pos_v,
        translate_h,
        translate_v,
        **params,
    ):
        m = tk.ndimage.compute_perspective(
            image.shape[1],
            image.shape[0],
            self.width,
            self.height,
            flip_h=flip_h,
            flip_v=flip_v,
            scale_h=scale_h,
            scale_v=scale_v,
            degrees=degrees,
            pos_h=pos_h,
            pos_v=pos_v,
            translate_h=translate_h,
            translate_v=translate_v,
        )
        raise tk.ndimage.transform_points(keypoint, m)

    def get_params(self, **data):
        scale = (
            self.base_scale
            * np.exp(
                data["random"].uniform(
                    np.log(self.scale_range[0]), np.log(self.scale_range[1])
                )
            )
            if data["random"].rand() <= self.scale_prob
            else self.base_scale
        )
        ar = (
            np.exp(
                data["random"].uniform(
                    np.log(self.aspect_range[0]), np.log(self.aspect_range[1])
                )
            )
            if data["random"].rand() <= self.aspect_prob
            else 1.0
        )
        pos_h, pos_v = data["random"].uniform(0, 1, size=2)
        translate_h = data["random"].uniform(-self.translate_h, self.translate_h)
        translate_v = data["random"].uniform(-self.translate_v, self.translate_v)
        return {
            "flip_h": self.flip_h and data["random"].rand() <= 0.5,
            "flip_v": self.flip_v and data["random"].rand() <= 0.5,
            "scale_h": scale * np.sqrt(ar),
            "scale_v": scale / np.sqrt(ar),
            "degrees": data["random"].uniform(
                self.rotate_range[0], self.rotate_range[1]
            )
            if data["random"].rand() <= self.rotate_prob
            else 0,
            "pos_h": pos_h,
            "pos_v": pos_v,
            "translate_h": translate_h,
            "translate_v": translate_v,
        }


class Resize(DualTransform):
    """リサイズ。"""

    def __init__(self, width, height, always_apply=False, p=1):
        super().__init__(always_apply=False, p=1)
        self.width = width
        self.height = height

    def apply(self, image, **params):
        return tk.ndimage.resize(image, width=self.width, height=self.height)

    def apply_to_bbox(self, bbox, **params):
        raise bbox

    def apply_to_keypoint(self, keypoint, **params):
        raise keypoint


class RandomColorAugmentors(RandomCompose):
    """色関連のDataAugmentationをいくつかまとめたもの。"""

    def __init__(self, p=1):
        argumentors = [
            GaussNoise(p=0.125),
            RandomBlur(p=0.125),
            RandomUnsharpMask(p=0.125),
            RandomSaturation(p=0.25),
            RandomBrightness(p=0.25),
            RandomContrast(p=0.25),
            RandomHue(p=0.25),
            RandomEqualize(p=0.0625),
            RandomAutoContrast(p=0.0625),
            RandomPosterize(p=0.0625),
            RandomAlpha(p=0.125),
        ]
        super().__init__(argumentors, p=p)


class GaussNoise(ImageOnlyTransform):
    """ガウシアンノイズ。"""

    def __init__(self, scale=5, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale

    def apply(self, image, random, **params):
        return tk.ndimage.gaussian_noise(image, random, self.scale)


class RandomBlur(ImageOnlyTransform):
    """ぼかし。"""

    def __init__(self, radius=0.75, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.radius = radius

    def apply(self, image, random, **params):
        return tk.ndimage.blur(image, self.radius * random.rand())


class RandomUnsharpMask(ImageOnlyTransform):
    """シャープ化。"""

    def __init__(self, sigma=0.5, min_alpha=1, max_alpha=2, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.sigma = sigma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def apply(self, image, random, **params):
        return tk.ndimage.unsharp_mask(
            image, self.sigma, random.uniform(self.min_alpha, self.max_alpha)
        )


class RandomBrightness(ImageOnlyTransform):
    """明度の変更。"""

    def __init__(self, shift=32, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.shift = shift

    def apply(self, image, random, **params):
        return tk.ndimage.brightness(image, random.uniform(-self.shift, self.shift))


class RandomContrast(ImageOnlyTransform):
    """コントラストの変更。"""

    def __init__(self, var=0.25, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.var = var

    def apply(self, image, random, **params):
        return tk.ndimage.contrast(image, random.uniform(1 - self.var, 1 + self.var))


class RandomSaturation(ImageOnlyTransform):
    """彩度の変更。"""

    def __init__(self, var=0.5, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.var = var

    def apply(self, image, random, **params):
        return tk.ndimage.saturation(image, random.uniform(1 - self.var, 1 + self.var))


class RandomHue(ImageOnlyTransform):
    """色相の変更。"""

    def __init__(self, var=1 / 16, shift=8, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.var = var
        self.shift = shift

    def apply(self, image, random, **params):
        alpha = random.uniform(1 - self.var, 1 + self.var, (3,))
        beta = random.uniform(-self.shift, +self.shift, (3,))
        return tk.ndimage.hue_lite(image, alpha, beta)


class RandomEqualize(ImageOnlyTransform):
    """ヒストグラム平坦化。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, image, **params):
        return tk.ndimage.equalize(image)


class RandomAutoContrast(ImageOnlyTransform):
    """オートコントラスト。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def apply(self, image, **params):
        return tk.ndimage.auto_contrast(image)


class RandomPosterize(ImageOnlyTransform):
    """ポスタリゼーション。

    ↓で有効そうだったので。

    ■AutoAugment: Learning Augmentation Policies from Data
    https://arxiv.org/abs/1805.09501

    """

    def __init__(self, min_bits=4, max_bits=7, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.min_bits = min_bits
        self.max_bits = max_bits

    def apply(self, image, random, **params):
        bits = random.randint(self.min_bits, self.max_bits + 1)
        return tk.ndimage.posterize(image, bits)


class RandomAlpha(ImageOnlyTransform):
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

    def apply(self, image, random, **params):
        return tk.ndimage.erase_random(
            image,
            random,
            bboxes=None,
            scale_low=self.scale_low,
            scale_high=self.scale_high,
            rate_1=self.rate_1,
            rate_2=self.rate_2,
            alpha=self.alpha,
            max_tries=self.max_tries,
        )


class RandomErasing(ImageOnlyTransform):
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

    def apply(self, image, random, bboxes=None, **params):
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
                if random.rand() <= self.object_aware_prob:
                    b = np.copy(b).astype(int)
                    # box内に含まれる他のboxを考慮
                    inter_boxes = np.copy(bboxes[inter[i]])
                    inter_boxes -= np.expand_dims(
                        np.tile(b[:2], 2), axis=0
                    )  # bに合わせて平行移動
                    # random erasing
                    image[b[1] : b[3], b[0] : b[2], :] = tk.ndimage.erase_random(
                        image[b[1] : b[3], b[0] : b[2], :],
                        random,
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
                random,
                bboxes=None,
                scale_low=self.scale_low,
                scale_high=self.scale_high,
                rate_1=self.rate_1,
                rate_2=self.rate_2,
                max_tries=self.max_tries,
            )
        return image


class Standardize(ImageOnlyTransform):
    """標準化。0～255に適当に収める。"""

    def apply(self, image, **params):
        return tk.ndimage.standardize(image)


class ToGrayScale(ImageOnlyTransform):
    """グレースケール化。チャンネル数はとりあえず維持。"""

    def apply(self, image, **params):
        assert len(image.shape) == 3
        start_shape = image.shape
        image = tk.ndimage.to_grayscale(image)
        image = np.tile(np.expand_dims(image, axis=-1), (1, 1, start_shape[-1]))
        assert image.shape == start_shape
        return image


class RandomBinarize(ImageOnlyTransform):
    """ランダム2値化(白黒化)。"""

    def __init__(
        self, threshold_min=128 - 32, threshold_max=128 + 32, always_apply=False, p=0.5
    ):
        super().__init__(always_apply=always_apply, p=p)
        assert 0 < threshold_min < 255
        assert 0 < threshold_max < 255
        assert threshold_min < threshold_max
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def apply(self, image, random, **params):
        threshold = random.uniform(self.threshold_min, self.threshold_max)
        return tk.ndimage.binarize(image, threshold)
