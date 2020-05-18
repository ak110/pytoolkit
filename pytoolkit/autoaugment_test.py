# pylint: disable=redefined-outer-name

import albumentations as A
import pytest

import pytoolkit as tk


@pytest.fixture()
def save_dir(check_dir):
    """結果の確認用"""
    d = check_dir / "autoaugment"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize(
    "policy,filename",
    [
        ("cifar10", "cifar.png"),
        ("cifar10", "Lenna.png"),
        ("svhn", "Lenna.png"),
        ("imagenet", "Lenna.png"),
    ],
)
def test_autoaugment(data_dir, save_dir, policy, filename):
    """画像の変換のテスト。目視したいので結果を`../___check/autoaugment/`に保存しちゃう。"""
    aug: A.Transforms = {
        "cifar10": tk.autoaugment.CIFAR10Policy,
        "svhn": tk.autoaugment.SVHNPolicy,
        "imagenet": tk.autoaugment.ImageNetPolicy,
    }[policy]()
    img_path = data_dir / filename
    img = tk.ndimage.load(img_path, grayscale=True)
    for i in range(32):
        augmented = aug(image=img)
        tk.ndimage.save(
            save_dir / f"{policy}.{img_path.stem}.{i}.png", augmented["image"]
        )


@pytest.mark.parametrize("grayscale, mag", [(False, 0), (False, 9), (True, 5)])
@pytest.mark.parametrize(
    "klass",
    [
        tk.autoaugment.ShearX,
        tk.autoaugment.ShearY,
        tk.autoaugment.TranslateX,
        tk.autoaugment.TranslateY,
        tk.autoaugment.Rotate,
        tk.autoaugment.AutoContrast,
        tk.autoaugment.Invert,
        tk.autoaugment.Equalize,
        tk.autoaugment.Solarize,
        tk.autoaugment.Posterize,
        tk.autoaugment.Contrast,
        tk.autoaugment.Color,
        tk.autoaugment.Brightness,
        tk.autoaugment.Sharpness,
    ],
)
def test_transforms(data_dir, save_dir, klass, grayscale, mag):
    """各変換の確認。"""
    img = tk.ndimage.load(data_dir / "cifar.png", grayscale=grayscale)
    img = klass(mag, p=1)(image=img)["image"]
    assert (img.ndim == 2 or img.shape[-1] == 1) == grayscale
    tk.ndimage.save(
        save_dir / f"transform.{klass.__name__}.{grayscale=}.{mag=}.png", img,
    )
