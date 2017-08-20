import pathlib
import shutil

import numpy as np
import PIL.Image
import sklearn.externals.joblib

import pytoolkit as tk


def test_filters():
    """画像の変換のテスト。目視したいので結果を`../___check/ndimage/`に保存しちゃう。"""
    base_dir = pathlib.Path(__file__).resolve().parent
    save_dir = base_dir.parent.joinpath('___check', 'ndimage')
    rand = np.random.RandomState(1234)
    filters = [
        ('original', lambda rgb: rgb),
        ('gaussian_noise', lambda rgb: tk.ndimage.gaussian_noise(rgb, rand, 5)),
        ('blur', lambda rgb: tk.ndimage.blur(rgb, 1)),
        ('unsharp_mask', lambda rgb: tk.ndimage.unsharp_mask(rgb, 1, 2)),
        ('median_2', lambda rgb: tk.ndimage.median(rgb, 2)),
        ('median_3', lambda rgb: tk.ndimage.median(rgb, 3)),
        ('saturation_075', lambda rgb: tk.ndimage.saturation(rgb, 0.75)),
        ('saturation_125', lambda rgb: tk.ndimage.saturation(rgb, 1.25)),
        ('brightness_075', lambda rgb: tk.ndimage.brightness(rgb, 0.75)),
        ('brightness_125', lambda rgb: tk.ndimage.brightness(rgb, 1.25)),
        ('contrast_075', lambda rgb: tk.ndimage.contrast(rgb, 0.75)),
        ('contrast_125', lambda rgb: tk.ndimage.contrast(rgb, 1.25)),
        ('lighting_pnn', lambda rgb: tk.ndimage.lighting(rgb, np.array([+1, -1, -1]))),
        ('lighting_n0p', lambda rgb: tk.ndimage.lighting(rgb, np.array([-1, +0, +1]))),
        ('lighting_ppp', lambda rgb: tk.ndimage.lighting(rgb, np.array([+1, +1, +1]))),
    ]

    rgb = tk.ndimage.load(base_dir.joinpath('data', 'Lenna.png'))
    save_dir.mkdir(parents=True, exist_ok=True)
    for cp in save_dir.iterdir():
        cp.unlink()
    for i, (name, filter_func) in enumerate(filters):
        x = filter_func(rgb)
        tk.ndimage.save(save_dir.joinpath('{:02d}_{}.png'.format(i, name)), x)
