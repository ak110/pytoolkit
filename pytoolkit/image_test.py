# pylint: disable=redefined-outer-name
import random

import albumentations as A
import numpy as np
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
    base_size = 32 if filename == "cifar.png" else 256
    bboxes = [(0.41, 0.39, 0.70, 0.75)]
    keypoints = [(0.52 * base_size, 0.52 * base_size)]
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
            # tk.image.RandomRotate(),  # TODO
            tk.image.RandomTransform(size=(256, 256), base_scale=2.0),
            tk.image.RandomColorAugmentors(noisy=True),
            tk.image.SpeckleNoise(),
            tk.image.GridMask(),
        ],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["classes"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )
    img_path = data_dir / filename
    original_img = tk.ndimage.load(img_path)
    for i in range(32):
        augmented = aug(
            image=original_img, bboxes=bboxes, classes=[0], keypoints=keypoints
        )
        img = augmented["image"]
        img = tk.od.plot_objects(img, None, None, augmented["bboxes"])
        for x, y in augmented["keypoints"]:
            x = np.clip(int(x), 1, img.shape[1] - 2)
            y = np.clip(int(y), 1, img.shape[1] - 2)
            img[range(y - 1, y + 2), x, :] = [[[255, 255, 0]]]
            img[y, range(x - 1, x + 2), :] = [[[255, 255, 0]]]
        tk.ndimage.save(save_dir / f"{img_path.stem}.DA.{i}.png", img)


def test_ToGrayScale(data_dir, save_dir):
    """ToGrayScale"""
    aug = tk.image.ToGrayScale(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    tk.ndimage.save(save_dir / "Lenna.ToGrayScale.png", aug(image=img)["image"])


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
            save_dir / f"Lenna.WrappedTranslateX.{s:+.2f}.png", aug.apply(img, scale=s)
        )


def test_WrappedTranslateY(data_dir, save_dir):
    """WrappedTranslateY"""
    aug = tk.image.WrappedTranslateY(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    for s in [-0.75, -0.25, +0.25, +0.75]:
        tk.ndimage.save(
            save_dir / f"Lenna.WrappedTranslateY.{s:+.2f}.png", aug.apply(img, scale=s)
        )


def test_PerlinNoise(data_dir, save_dir):
    """PerlinNoise"""
    aug = tk.image.PerlinNoise(p=1)
    img = tk.ndimage.load(data_dir / "Lenna.png")
    for i in range(10):
        tk.ndimage.save(
            save_dir / f"Lenna.PerlinNoise.{i}.png", aug(image=img)["image"]
        )


def test_gray_scale(data_dir):
    img = tk.ndimage.load(data_dir / "Lenna.png", grayscale=True)
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
            tk.image.RandomTransform(size=(256, 256)),
            tk.image.RandomColorAugmentors(noisy=True, grayscale=True),
            tk.image.SpeckleNoise(),
        ]
    )
    for _ in range(32):
        img = aug(image=img)["image"]
        assert img.shape == (256, 256, 1)


def test_RandomTransform_with_bboxes():
    random.seed(1)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    # 左上と右下に1個ずつboxを用意
    bboxes = np.array([[0, 0, 1, 1], [31, 31, 32, 32]]) / 32.0
    classes = ["A", "B"]
    aug = A.Compose(
        [tk.image.RandomTransform(size=(8, 8), base_scale=2.0, with_bboxes=True)],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["classes"]),
    )
    a_count = 0
    b_count = 0
    for _ in range(300):
        d = aug(image=image, bboxes=bboxes, classes=classes)
        assert d["image"].shape == (8, 8, 3)
        if len(d["bboxes"]) == 0:
            assert len(d["classes"]) == 0
        else:
            assert len(d["bboxes"]) in (1, 2)
            if "A" in d["classes"]:
                a_count += 1
            if "B" in d["classes"]:
                b_count += 1
    # 100回中何回bboxが出力されたか。
    # (挙動が変わった時に気付きやすいように==にしているが、
    #  with_bboxes=Falseの場合(≒(0, 0))より多く、かつa_count ≒ b_countになればOK)
    assert (a_count, b_count) == (104, 104)


@pytest.mark.parametrize("resize", ["normal", "preserve_aspect"])
def test_RandomTransform_crop1(resize):
    # translateとrotateを無効にすれば出力にpaddingが含まれないことの確認 (normal, preserve_aspect)
    random.seed(1)
    image = np.ones((32, 30, 3), dtype=np.uint8)
    aug = tk.image.RandomTransform(
        size=(8, 8),  # 40x40くらいに拡大して8x8でcrop
        base_scale=5.0,
        border_mode="zero",
        rotate_prob=0.0,
        translate=(0.0, 0.0),
        mode=resize,
    )
    for i in range(100):
        augmented_image = aug(image=image)["image"]
        assert (augmented_image == 1).all(), f"{i=} {augmented_image=}"


@pytest.mark.parametrize("base_scale", [0.75, 1.0, 1.5])
def test_RandomTransform_crop2(base_scale):
    # translateとrotateを無効にすれば出力にpaddingが含まれないことの確認 (preserve_pixel)
    random.seed(1)
    image = np.ones((64, 60, 3), dtype=np.uint8)
    aug = tk.image.RandomTransform(
        size=(8, 8),  # 48～96四方くらい(変動あり)から8x8でcrop
        base_scale=base_scale,
        border_mode="zero",
        rotate_prob=0.0,
        translate=(0.0, 0.0),
        mode="preserve_pixel",
    )
    for i in range(100):
        augmented_image = aug(image=image)["image"]
        assert (augmented_image == 1).all(), f"{i=} {augmented_image=}"


@pytest.mark.parametrize("resize", ["normal", "preserve_aspect"])
def test_RandomTransform_pad1(resize):
    # パディングされていることの確認 (normal, preserve_aspect)
    random.seed(1)
    image = np.ones((32, 30, 3), dtype=np.uint8)
    aug = tk.image.RandomTransform(
        size=(16, 16),  # 4x4くらいに縮小してpadding混みで出力
        base_scale=0.25,
        border_mode="zero",
        translate=(0.0, 0.0),
        mode=resize,
    )
    augmented_image = aug(image=image)["image"]
    assert not (augmented_image == 1).all()
    assert (augmented_image == 1).any()
    # 四辺は全てpaddingのはず
    assert (augmented_image[:1, :, :] == 0).all()
    assert (augmented_image[-1:, :, :] == 0).all()
    assert (augmented_image[:, :1, :] == 0).all()
    assert (augmented_image[:, -1:, :] == 0).all()


@pytest.mark.parametrize("base_scale", [0.75, 1.0, 1.5])
def test_RandomTransform_pad2(base_scale):
    # パディングされていることの確認 (preserve_pixel)
    random.seed(1)
    image = np.ones((32, 30, 3), dtype=np.uint8)
    aug = tk.image.RandomTransform(
        size=(100, 100),  # 24～48四方くらい(変動あり)から100px四方にパディング
        base_scale=base_scale,
        border_mode="zero",
        translate=(0.0, 0.0),
        mode="preserve_pixel",
    )
    augmented_image = aug(image=image)["image"]
    assert not (augmented_image == 1).all()
    assert (augmented_image == 1).any()
    # 四辺は全てpaddingのはず
    edge_l = (augmented_image[:1, :, :] == 0).all()
    edge_r = (augmented_image[-1:, :, :] == 0).all()
    edge_t = (augmented_image[:, :1, :] == 0).all()
    edge_b = (augmented_image[:, -1:, :] == 0).all()
    assert edge_l or edge_r
    assert edge_t or edge_b
