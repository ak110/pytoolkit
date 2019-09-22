# pylint: disable=redefined-outer-name
import albumentations as A
import pytest

import pytoolkit as tk


@pytest.fixture()
def save_dir(check_dir):
    """結果の確認用"""
    d = check_dir / "image"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize("filename", ["cifar.png", "Lenna.png"])
def test_data_augmentation(data_dir, save_dir, filename):
    """画像の変換のテスト。目視したいので結果を`../___check/image/`に保存しちゃう。"""
    aug = A.Compose(
        [
            A.OneOf(
                [
                    tk.image.Standardize(),
                    tk.image.ToGrayScale(p=0.125),
                    tk.image.RandomBinarize(p=0.125),
                ],
                p=0.25,
            ),
            tk.image.RandomRotate(),
            tk.image.RandomTransform(256, 256),
            tk.image.RandomColorAugmentors(noisy=True),
            tk.image.SpeckleNoise(),
        ]
    )
    img_path = data_dir / filename
    img = tk.ndimage.load(img_path)
    for i in range(32):
        tk.ndimage.save(
            save_dir / f"{img_path.stem}.DA.{i}.png", aug(image=img)["image"]
        )


def test_ToGrayScale(data_dir, save_dir):
    """ToGrayScale"""
    aug = tk.image.ToGrayScale(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    tk.ndimage.save(save_dir / f"Lenna.ToGrayScale.png", aug(image=img)["image"])


def test_RandomBinarize(data_dir, save_dir):
    """RandomBinarize"""
    aug = tk.image.RandomBinarize(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    for i in range(4):
        tk.ndimage.save(
            save_dir / f"Lenna.RandomBinarize.{i}.png", aug(image=img)["image"]
        )


def test_WrappedTranslateX(data_dir, save_dir):
    """WrappedTranslateX"""
    aug = tk.image.WrappedTranslateX(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    for s in [-0.75, -0.25, +0.25, +0.75]:
        tk.ndimage.save(
            save_dir / f"Lenna.WrappedTranslateX.{s:+.2f}.png",
            aug.apply(image=img, scale=s),
        )


def test_WrappedTranslateY(data_dir, save_dir):
    """WrappedTranslateY"""
    aug = tk.image.WrappedTranslateY(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    for s in [-0.75, -0.25, +0.25, +0.75]:
        tk.ndimage.save(
            save_dir / f"Lenna.WrappedTranslateY.{s:+.2f}.png",
            aug.apply(image=img, scale=s),
        )
