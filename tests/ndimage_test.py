import pathlib

import numpy as np

import pytoolkit as tk


def test_filters():
    """画像の変換のテスト。目視したいので結果を`../___check/ndimage/`に保存しちゃう。"""
    base_dir = pathlib.Path(__file__).resolve().parent
    save_dir = base_dir.parent.joinpath('___check', 'ndimage')
    rand = np.random.RandomState(1234)
    filters = [
        (0, 'original', lambda rgb: rgb),
        (0, 'pad_same', lambda rgb: tk.ndimage.pad(rgb, 300, 300)),
        (0, 'pad_zero', lambda rgb: tk.ndimage.pad(rgb, 300, 300, padding='zero')),
        (0, 'pad_refl', lambda rgb: tk.ndimage.pad(rgb, 300, 300, padding='reflect')),
        (0, 'pad_wrap', lambda rgb: tk.ndimage.pad(rgb, 300, 300, padding='wrap')),
        (0, 'pad_rand', lambda rgb: tk.ndimage.pad(rgb, 300, 300, padding='rand', rand=np.random)),
        (0, 'rotate', lambda rgb: tk.ndimage.rotate(rgb, 15)),
        (0, 'rotate', lambda rgb: tk.ndimage.rotate(rgb, -15)),
        (0, 'rotate_z', lambda rgb: tk.ndimage.rotate(rgb, -15, padding='zero')),
        (0, 'rotate_r', lambda rgb: tk.ndimage.rotate(rgb, -15, padding='reflect')),
        (0, 'rotate_w', lambda rgb: tk.ndimage.rotate(rgb, -15, padding='wrap')),
        (0, 'crop', lambda rgb: tk.ndimage.crop(rgb, 30, 30, 200, 200)),
        (0, 'flip_lr', tk.ndimage.flip_lr),
        (0, 'flip_tb', tk.ndimage.flip_tb),
        (0, 'resize', lambda rgb: tk.ndimage.resize(rgb, 128, 64, padding='same')),
        (0, 'resize', lambda rgb: tk.ndimage.resize(rgb, 128, 64, padding=None)),
        (1, 'gaussian_noise', lambda rgb: tk.ndimage.gaussian_noise(rgb, rand, 5)),
        (1, 'blur', lambda rgb: tk.ndimage.blur(rgb, 0.5)),
        (1, 'unsharp_mask', lambda rgb: tk.ndimage.unsharp_mask(rgb, 0.5, 1.5)),
        (1, 'median_2', lambda rgb: tk.ndimage.median(rgb, 2)),
        (1, 'median_3', lambda rgb: tk.ndimage.median(rgb, 3)),
        (1, 'brightness_n', lambda rgb: tk.ndimage.brightness(rgb, -32)),
        (1, 'brightness_p', lambda rgb: tk.ndimage.brightness(rgb, 32)),
        (1, 'saturation_l', lambda rgb: tk.ndimage.saturation(rgb, 0.5)),
        (1, 'saturation_h', lambda rgb: tk.ndimage.saturation(rgb, 1.5)),
        (1, 'contrast_l', lambda rgb: tk.ndimage.contrast(rgb, 0.75)),
        (1, 'contrast_h', lambda rgb: tk.ndimage.contrast(rgb, 1.25)),
        (1, 'hue_n', lambda rgb: tk.ndimage.hue(rgb, -0.1)),
        (1, 'hue_p', lambda rgb: tk.ndimage.hue(rgb, +0.1)),
    ]

    rgb = tk.ndimage.load(base_dir.joinpath('data', 'Lenna.png'))  # 256x256の某有名画像
    save_dir.mkdir(parents=True, exist_ok=True)
    for cp in save_dir.iterdir():
        cp.unlink()
    for i, (partial, name, filter_func) in enumerate(filters):
        if partial:
            x = np.copy(rgb)
            x[64:-64, 64:-64, :] = filter_func(x[64:-64, 64:-64, :])
        else:
            x = filter_func(rgb)
        tk.ndimage.save(save_dir.joinpath('{:02d}_{}.png'.format(i, name)), x)
