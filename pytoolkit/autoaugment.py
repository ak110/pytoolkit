"""AutoAugment <https://arxiv.org/abs/1805.09501>らしきもの。

<https://github.com/tensorflow/models/tree/master/research/autoaugment>

"""
# pylint: disable=arguments-differ,abstract-method,unused-argument
import random

import albumentations as A
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps


class CIFAR10Policy(A.OneOf):
    """AutoAugment <https://arxiv.org/abs/1805.09501>のCIFAR-10用。"""

    def __init__(self, p=1):
        super().__init__(
            [
                subpolicy(Invert, 0.1, 7, Contrast, 0.2, 6),
                subpolicy(Rotate, 0.7, 2, TranslateX, 0.3, 9),
                subpolicy(Sharpness, 0.8, 1, Sharpness, 0.9, 3),
                subpolicy(ShearY, 0.5, 8, TranslateY, 0.7, 9),
                subpolicy(AutoContrast, 0.5, 8, Equalize, 0.9, 2),
                subpolicy(ShearY, 0.2, 7, Posterize, 0.3, 7),
                subpolicy(Color, 0.4, 3, Brightness, 0.6, 7),
                subpolicy(Sharpness, 0.3, 9, Brightness, 0.7, 9),
                subpolicy(Equalize, 0.6, 5, Equalize, 0.5, 1),
                subpolicy(Contrast, 0.6, 7, Sharpness, 0.6, 5),
                subpolicy(Color, 0.7, 7, TranslateX, 0.5, 8),
                subpolicy(Equalize, 0.3, 7, AutoContrast, 0.4, 8),
                subpolicy(TranslateY, 0.4, 3, Sharpness, 0.2, 6),
                subpolicy(Brightness, 0.9, 6, Color, 0.2, 8),
                subpolicy(Solarize, 0.5, 2, Invert, 0.0, 3),
                subpolicy(Equalize, 0.2, 0, AutoContrast, 0.6, 0),
                subpolicy(Equalize, 0.2, 8, Equalize, 0.6, 4),
                subpolicy(Color, 0.9, 9, Equalize, 0.6, 6),
                subpolicy(AutoContrast, 0.8, 4, Solarize, 0.2, 8),
                subpolicy(Brightness, 0.1, 3, Color, 0.7, 0),
                subpolicy(Solarize, 0.4, 5, AutoContrast, 0.9, 3),
                subpolicy(TranslateY, 0.9, 9, TranslateY, 0.7, 9),
                subpolicy(AutoContrast, 0.9, 2, Solarize, 0.8, 3),
                subpolicy(Equalize, 0.8, 8, Invert, 0.1, 3),
                subpolicy(TranslateY, 0.7, 9, AutoContrast, 0.9, 1),
            ],
            p=p,
        )

    def get_transform_init_args_names(self):
        return ()


class SVHNPolicy(A.OneOf):
    """AutoAugment <https://arxiv.org/abs/1805.09501>のSVHN用。"""

    def __init__(self, p=1):
        super().__init__(
            [
                subpolicy(ShearX, 0.9, 4, Invert, 0.2, 3),
                subpolicy(ShearY, 0.9, 8, Invert, 0.7, 5),
                subpolicy(Equalize, 0.6, 5, Solarize, 0.6, 6),
                subpolicy(Invert, 0.9, 3, Equalize, 0.6, 3),
                subpolicy(Equalize, 0.6, 1, Rotate, 0.9, 3),
                subpolicy(ShearX, 0.9, 4, AutoContrast, 0.8, 3),
                subpolicy(ShearY, 0.9, 8, Invert, 0.4, 5),
                subpolicy(ShearY, 0.9, 5, Solarize, 0.2, 6),
                subpolicy(Invert, 0.9, 6, AutoContrast, 0.8, 1),
                subpolicy(Equalize, 0.6, 3, Rotate, 0.9, 3),
                subpolicy(ShearX, 0.9, 4, Solarize, 0.3, 3),
                subpolicy(ShearY, 0.8, 8, Invert, 0.7, 4),
                subpolicy(Equalize, 0.9, 5, TranslateY, 0.6, 6),
                subpolicy(Invert, 0.9, 4, Equalize, 0.6, 7),
                subpolicy(Contrast, 0.3, 3, Rotate, 0.8, 4),
                subpolicy(Invert, 0.8, 5, TranslateY, 0.0, 2),
                subpolicy(ShearY, 0.7, 6, Solarize, 0.4, 8),
                subpolicy(Invert, 0.6, 4, Rotate, 0.8, 4),
                subpolicy(ShearY, 0.3, 7, TranslateX, 0.9, 3),
                subpolicy(ShearX, 0.1, 6, Invert, 0.6, 5),
                subpolicy(Solarize, 0.7, 2, TranslateY, 0.6, 7),
                subpolicy(ShearY, 0.8, 4, Invert, 0.8, 8),
                subpolicy(ShearX, 0.7, 9, TranslateY, 0.8, 3),
                subpolicy(ShearY, 0.8, 5, AutoContrast, 0.7, 3),
                subpolicy(ShearX, 0.7, 2, Invert, 0.1, 5),
            ],
            p=p,
        )

    def get_transform_init_args_names(self):
        return ()


class ImageNetPolicy(A.OneOf):
    """AutoAugment <https://arxiv.org/abs/1805.09501>のImageNet用。"""

    def __init__(self, p=1):
        super().__init__(
            [
                subpolicy(Posterize, 0.4, 8, Rotate, 0.6, 9),
                subpolicy(Solarize, 0.6, 5, AutoContrast, 0.6, 5),
                subpolicy(Equalize, 0.8, 8, Equalize, 0.6, 3),
                subpolicy(Posterize, 0.6, 7, Posterize, 0.6, 6),
                subpolicy(Equalize, 0.4, 7, Solarize, 0.2, 4),
                subpolicy(Equalize, 0.4, 4, Rotate, 0.8, 8),
                subpolicy(Solarize, 0.6, 3, Equalize, 0.6, 7),
                subpolicy(Posterize, 0.8, 5, Equalize, 1.0, 2),
                subpolicy(Rotate, 0.2, 3, Solarize, 0.6, 8),
                subpolicy(Equalize, 0.6, 8, Posterize, 0.4, 6),
                subpolicy(Rotate, 0.8, 8, Color, 0.4, 0),
                subpolicy(Rotate, 0.4, 9, Equalize, 0.6, 2),
                subpolicy(Equalize, 0.0, 7, Equalize, 0.8, 8),
                subpolicy(Invert, 0.6, 4, Equalize, 1.0, 8),
                subpolicy(Color, 0.6, 4, Contrast, 1.0, 8),
                subpolicy(Rotate, 0.8, 8, Color, 1.0, 2),
                subpolicy(Color, 0.8, 8, Solarize, 0.8, 7),
                subpolicy(Sharpness, 0.4, 7, Invert, 0.6, 8),
                subpolicy(ShearX, 0.6, 5, Equalize, 1.0, 9),
                subpolicy(Color, 0.4, 0, Equalize, 0.6, 3),
                subpolicy(Equalize, 0.4, 7, Solarize, 0.2, 4),
                subpolicy(Solarize, 0.6, 5, AutoContrast, 0.6, 5),
                subpolicy(Invert, 0.6, 4, Equalize, 1.0, 8),
                subpolicy(Color, 0.6, 4, Contrast, 1.0, 8),
                subpolicy(Equalize, 0.8, 8, Equalize, 0.6, 3),
            ],
            p=p,
        )

    def get_transform_init_args_names(self):
        return ()


def subpolicy(a1, p1, mag1, a2, p2, mag2):
    """サブポリシー。"""
    return A.Compose([a1(mag=mag1, p=p1), a2(mag=mag2, p=p2)], p=1)


class Affine(A.ImageOnlyTransform):
    """Affine変換。

    TODO: 物体検出とかへの対応。

    """

    def __init__(
        self,
        shear_x_mag=0,
        shear_y_mag=0,
        translate_x_mag=0,
        translate_y_mag=0,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.shear_x_mag = shear_x_mag
        self.shear_y_mag = shear_y_mag
        self.translate_x_mag = translate_x_mag
        self.translate_y_mag = translate_y_mag

    def apply(self, image, **params):
        shear_x = float_parameter(self.shear_x_mag, 0.3, flip_sign=True)
        shear_y = float_parameter(self.shear_y_mag, 0.3, flip_sign=True)
        translate_x = float_parameter(
            self.translate_x_mag, image.shape[1] * 150 / 331, flip_sign=True
        )
        translate_y = float_parameter(
            self.translate_y_mag, image.shape[0] * 150 / 331, flip_sign=True
        )
        image = to_pillow(image)
        data = (1, shear_x, translate_x, shear_y, 1, translate_y)
        return np.asarray(
            image.transform(
                image.size,
                PIL.Image.AFFINE,
                data,
                PIL.Image.BICUBIC,
                fillcolor=(128,) if image.mode == "L" else (128, 128, 128),
            ),
            dtype=np.uint8,
        )

    def get_transform_init_args_names(self):
        return (
            "shear_x_mag",
            "shear_y_mag",
            "translate_x_mag",
            "translate_y_mag",
        )


class ShearX(Affine):
    """せん断。"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(shear_x_mag=mag, always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ("mag",)


class ShearY(Affine):
    """せん断。"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(shear_y_mag=mag, always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ("mag",)


class TranslateX(Affine):
    """移動。"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(translate_x_mag=mag, always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ("mag",)


class TranslateY(Affine):
    """移動。"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(translate_y_mag=mag, always_apply=always_apply, p=p)

    def get_transform_init_args_names(self):
        return ("mag",)


class Rotate(A.ImageOnlyTransform):
    """回転。

    TODO: maskとかへの対応。

    """

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        degrees = int_parameter(self.mag, 30, flip_sign=True)
        image = to_pillow(image)
        in_mode = image.mode
        image = image.convert("RGBA").rotate(degrees)
        bg = PIL.Image.new("RGBA", image.size, (128, 128, 128, 255))
        image = PIL.Image.composite(image, bg, image).convert(in_mode)
        return np.asarray(image, dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class AutoContrast(A.ImageOnlyTransform):
    """PIL.ImageOps.autocontrastなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        del mag

    def apply(self, image, **params):
        image = to_pillow(image)
        return np.asarray(PIL.ImageOps.autocontrast(image), dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class Invert(A.ImageOnlyTransform):
    """PIL.ImageOps.equalizeなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        del mag

    def apply(self, image, **params):
        image = to_pillow(image)
        return np.asarray(PIL.ImageOps.invert(image), dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class Equalize(A.ImageOnlyTransform):
    """PIL.ImageOps.equalizeなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        del mag

    def apply(self, image, **params):
        image = to_pillow(image)
        return np.asarray(PIL.ImageOps.equalize(image), dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class Solarize(A.ImageOnlyTransform):
    """PIL.ImageOps.solarizeなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        threshold = 256 - int_parameter(self.mag, 256)
        image = to_pillow(image)
        return np.asarray(PIL.ImageOps.solarize(image, threshold), dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class Posterize(A.ImageOnlyTransform):
    """PIL.ImageOps.posterizeなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        # https://github.com/tensorflow/models/blob/master/research/autoaugment/augmentation_transforms.py#L267 🤔
        bit = 8 - int_parameter(self.mag, 4)
        image = to_pillow(image)
        return np.asarray(PIL.ImageOps.posterize(image, bit), dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class Contrast(A.ImageOnlyTransform):
    """PIL.ImageEnhance.ContrastなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        factor = 1 + float_parameter(self.mag, 0.9, flip_sign=True)
        image = to_pillow(image)
        return np.asarray(
            PIL.ImageEnhance.Contrast(image).enhance(factor), dtype=np.uint8
        )

    def get_transform_init_args_names(self):
        return ("mag",)


class Color(A.ImageOnlyTransform):
    """PIL.ImageEnhance.ColorなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        factor = 1 + float_parameter(self.mag, 0.9, flip_sign=True)
        image = to_pillow(image)
        return np.asarray(PIL.ImageEnhance.Color(image).enhance(factor), dtype=np.uint8)

    def get_transform_init_args_names(self):
        return ("mag",)


class Brightness(A.ImageOnlyTransform):
    """PIL.ImageEnhance.BrightnessなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        factor = 1 + float_parameter(self.mag, 0.9, flip_sign=True)
        image = to_pillow(image)
        return np.asarray(
            PIL.ImageEnhance.Brightness(image).enhance(factor), dtype=np.uint8
        )

    def get_transform_init_args_names(self):
        return ("mag",)


class Sharpness(A.ImageOnlyTransform):
    """PIL.ImageEnhance.SharpnessなTransform"""

    def __init__(self, mag, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.mag = mag

    def apply(self, image, **params):
        factor = 1 + float_parameter(self.mag, 0.9, flip_sign=True)
        image = to_pillow(image)
        return np.asarray(
            PIL.ImageEnhance.Sharpness(image).enhance(factor), dtype=np.uint8
        )

    def get_transform_init_args_names(self):
        return ("mag",)


def to_pillow(image: np.ndarray) -> PIL.Image:
    """ndarrayからPIL.Imageへの変換。"""
    image = np.squeeze(image)
    mode = "L" if image.ndim == 2 else "RGB"
    return PIL.Image.fromarray(image, mode)


def float_parameter(level: int, maxval: float, flip_sign: bool = False) -> float:
    """0～maxvalへの変換。"""
    assert 0 <= level <= 9
    value = float(level) * maxval / 9
    if flip_sign and random.random() < 0.5:
        value = -value
    return value


def int_parameter(level: int, maxval: int, flip_sign: bool = False) -> int:
    """0～maxvalへの変換。"""
    assert 0 <= level <= 9
    value = int(np.round(level * maxval / 9))
    if flip_sign and random.random() < 0.5:
        value = -value
    return value
