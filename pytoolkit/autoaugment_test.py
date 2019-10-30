# pylint: disable=redefined-outer-name
import pytest

import pytoolkit as tk


@pytest.fixture()
def save_dir(check_dir):
    """結果の確認用"""
    d = check_dir / "autoaugment"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.parametrize("filename", ["cifar.png", "Lenna.png"])
def test_autoaugment(data_dir, save_dir, filename):
    """画像の変換のテスト。目視したいので結果を`../___check/autoaugment/`に保存しちゃう。"""
    aug = tk.autoaugment.CIFAR10Policy()
    img_path = data_dir / filename
    img = tk.ndimage.load(img_path)
    for i in range(32):
        tk.ndimage.save(save_dir / f"{img_path.stem}.{i}.png", aug(image=img)["image"])


@pytest.mark.parametrize("mag", [0, 9])
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
def test_transforms(data_dir, save_dir, klass, mag):
    """各変換の確認。"""
    img = tk.ndimage.load(data_dir / "cifar.png")
    img = klass(mag, p=1)(image=img)["image"]
    tk.ndimage.save(save_dir / f"transform.{klass.__name__}.mag={mag}.png", img)
